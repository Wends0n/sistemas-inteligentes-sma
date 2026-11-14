/**
 * @file sma.c
 * @brief Núcleo do Slime Mould Algorithm (SMA) para minimização.
 *
 * Implementa a versão da demo dos autores (SMA.m, v1.0, 2020) a partir de
 * Li et al., Future Generation Computer Systems 111 (2020) 300-323.
 * O módulo é genérico: não conhece o PFS. O problema entra por callback
 * (`Objective`), o que mantém o solver separado do problema (ver pfs.c).
 *
 * Convenção: menor aptidão é melhor. Cada agente é uma linha da matriz x
 * (N x D, armazenada por linhas): a coordenada j do agente i é x[i*D + j].
 */
#include "sma.h"
#include "rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * Peso W do agente na posição `rank` do ranking (base zero, 0 = melhor).
 *
 *   q = (s - melhor) / (pior - melhor)            em [0, 1]
 *   W = 1 + u * log10(1 + q)   primeira metade do ranking (melhores)
 *   W = 1 - u * log10(1 + q)   segunda metade   (piores)
 *
 * u ~ U[0,1) é sorteado pelo chamador, um por coordenada.
 * Se todos empatam (pior == melhor), adota-se q = 0, logo W = 1. A demo
 * MATLAB evita a divisão por zero somando `eps` ao denominador; aqui o
 * desvio é explícito. `rank < n/2` (divisão inteira) equivale ao
 * `i <= N/2` do MATLAB, com índice em base um, também para N ímpar.
 */
double sma_weight(double s, double best, double worst, int rank, int n, double u) {
    double q = (worst == best) ? 0.0 : (s - best) / (worst - best);
    double sinal = (rank < n / 2) ? 1.0 : -1.0;
    return 1.0 + sinal * u * log10(1.0 + q);
}

/**
 * Executa o SMA. Veja o contrato completo em sma.h.
 *
 * Ordem das operações em cada iteração t = 1..T:
 *   1. projeta cada agente em [lower, upper] e o avalia (N chamadas);
 *   2. ordena os agentes por aptidão (menor primeiro);
 *   3. calcula os pesos W (um sorteio por agente e coordenada);
 *   4. atualiza o melhor histórico se houve melhora estrita;
 *   5. move os agentes (reinicialização, aproximação ou contração);
 *   6. registra o melhor histórico em history[t-1].
 * A movimentação da última iteração não é avaliada (igual à demo original),
 * logo o total é exatamente N*T chamadas ao objetivo.
 *
 * @return 0 em sucesso; -1 se parâmetros inválidos, falta de memória ou
 *         aptidão/coordenada não finita.
 */
static double clamp01(double v) { return fmin(1.0, fmax(0.0, v)); }

/* Controlador fuzzy Takagi-Sugeno de ordem zero (contrato em sma.h).
 * Regras: R1 (D baixa, E baixa), R2 (D baixa, E alta), R3 (D alta, E baixa),
 * R4 (D alta, E alta). Diversidade baixa e estagnação alta pedem mais
 * reinicialização (exploração); diversidade alta e estagnação baixa pedem
 * menos (refinamento). Os singletons padrão vêm de scripts/fuzzy_optuna.py. */
static double ramp_down(double v, const double r[2]) { return clamp01((r[1] - v) / (r[1] - r[0])); }
static double ramp_up(double v, const double r[2])   { return clamp01((v - r[0]) / (r[1] - r[0])); }

void sma_fuzzy_defaults(FuzzyParams *fp) {
    /* Calibração: results/fuzzy/calibracao_optuna.json. */
    *fp = (FuzzyParams){
        .div_low  = {0.056, 0.132}, .div_high = {0.108, 0.198},
        .stag_low = {0.014, 0.537}, .stag_high = {0.204, 1.000},
        .out = {0.0325, 0.0681, 0.012, 0.0619},   /* R1, R2, R3, R4 */
        .stag_window = 0.49
    };}

double sma_fuzzy_z(const FuzzyParams *fp, double diversity, double stagnation) {
    double dl = ramp_down(diversity, fp->div_low),  dh = ramp_up(diversity, fp->div_high);
    double sl = ramp_down(stagnation, fp->stag_low), sh = ramp_up(stagnation, fp->stag_high);
    double w[4] = { dl * sl, dl * sh, dh * sl, dh * sh };
    double num = 0.0, den = 0.0;
    for (int i = 0; i < 4; ++i) { num += w[i] * fp->out[i]; den += w[i]; }
    return den > 1e-12 ? num / den : 0.04;
}

static int sma_solve_impl(const SmaConfig *c, Objective f, void *ctx,
              double *best_x, double *history, double *best_f,
              const FuzzyParams *fp, double *z_history,
              double *div_history, double *stag_history)
{
    const int adaptive = fp != NULL;
    FuzzyParams dflt; sma_fuzzy_defaults(&dflt);   /* janela de estagnação nos traços */
    const double stag_window = adaptive ? fp->stag_window : dflt.stag_window;
    /* ---- Validação de parâmetros (falha cedo, nada é alocado) ---- */
    if (!c || !f || !best_x || !history || !best_f ||
        c->population < 1 || c->population > 10000 ||
        c->dimension  < 1 || c->dimension  > 10000 ||
        c->iterations < 1 ||
        !isfinite(c->lower) || !isfinite(c->upper) ||
        c->lower >= c->upper || !isfinite(c->upper - c->lower) ||
        !isfinite(c->z) || c->z < 0 || c->z > 1)
        return -1;

    const int n = c->population;   /* N: número de agentes (mofos)       */
    const int d = c->dimension;    /* D: coordenadas por agente           */
    int status = -1;               /* só vira 0 ao terminar com sucesso   */
    size_t count = (size_t)n * (size_t)d;

    /* ---- Memória de trabalho ---- */
    double *x    = malloc(count * sizeof *x);       /* posições (N x D)           */
    double *w    = malloc(count * sizeof *w);       /* pesos W (N x D)            */
    double *s    = malloc((size_t)n * sizeof *s);   /* aptidão de cada agente     */
    double *vb   = malloc((size_t)d * sizeof *vb);  /* vb ~ U(-a, a), por agente  */
    double *vc   = malloc((size_t)d * sizeof *vc);  /* vc ~ U(-b, b), por agente  */
    int    *rank = malloc((size_t)n * sizeof *rank);/* rank[k] = índice do k-ésimo melhor */
    if (!x || !w || !s || !vb || !vc || !rank) goto done;

    Rng rng = { c->seed };         /* gerador local: execuções não interferem */
    *best_f = INFINITY;

    /* ---- Inicialização: cada coordenada uniforme em [lower, upper) ---- */
    for (size_t k = 0; k < count; k++)
        x[k] = c->lower + (c->upper - c->lower) * rng_uniform(&rng);

    int stagnant = 0;
    for (int t = 1; t <= c->iterations; t++) {

        /* ---- 1 e 2. Projeção, avaliação e ranking ---- */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                size_t k = (size_t)i * d + j;
                if (!isfinite(x[k])) goto done;
                /* Projeção nas fronteiras (clamp) antes de avaliar. */
                x[k] = fmin(c->upper, fmax(c->lower, x[k]));
            }
            s[i] = f(x + (size_t)i * d, d, ctx);   /* uma avaliação por agente */
            if (!isfinite(s[i])) goto done;

            /* Inserção do agente i no ranking parcial rank[0..i-1].
             * É estável: em empate, o agente de menor índice fica na frente. */
            int k = i;
            while (k > 0 && s[rank[k - 1]] > s[i]) { rank[k] = rank[k - 1]; k--; }
            rank[k] = i;
        }
        const double best  = s[rank[0]];       /* melhor aptidão desta iteração */
        const double worst = s[rank[n - 1]];   /* pior aptidão desta iteração   */

        /* ---- 3. Pesos W: um u ~ U[0,1) por agente e coordenada ---- */
        for (int k = 0; k < n; k++)
            for (int j = 0; j < d; j++)
                w[(size_t)rank[k] * d + j] =
                    sma_weight(s[rank[k]], best, worst, k, n, rng_uniform(&rng));

        /* ---- 4. Melhor histórico (copia o vetor, não guarda ponteiro) ----
         * best_x é uma cópia separada de x, que será alterado no passo 5. */
        if (best < *best_f) {
            *best_f = best;
            memcpy(best_x, x + (size_t)rank[0] * d, (size_t)d * sizeof *x);
            stagnant = 0;
        } else {
            stagnant++;
        }

        double z_current = c->z;
        if (adaptive || div_history || stag_history) {
            double diversity = 0.0;
            for (int j = 0; j < d; ++j) {
                double mean = 0.0, var = 0.0;
                for (int i = 0; i < n; ++i) mean += x[(size_t)i * d + j];
                mean /= n;
                for (int i = 0; i < n; ++i) {
                    double delta = x[(size_t)i * d + j] - mean;
                    var += delta * delta;
                }
                var /= n;
                diversity += sqrt(var) / (c->upper - c->lower);
            }
            diversity = clamp01(diversity / d);
            double stagnation = clamp01((double)stagnant / fmax(10.0, stag_window * c->iterations));
            if (adaptive) z_current = sma_fuzzy_z(fp, diversity, stagnation);
            if (div_history)  div_history[t - 1]  = diversity;
            if (stag_history) stag_history[t - 1] = stagnation;
        }
        if (z_history) z_history[t - 1] = z_current;

        /* Parâmetros que decaem ao longo das iterações:
         *   b = 1 - t/T   vai de ~1 a 0 (amplitude da contração)
         *   a = atanh(b)  vai de grande a 0 (amplitude da aproximação) */
        const double b = 1.0 - (double)t / c->iterations;
        const double a = atanh(b);

        /* ---- 5. Movimentação ----
         * A matriz x é alterada no lugar e em sequência: um agente já movido
         * pode ser sorteado como A ou B por agentes seguintes. Já s, w e
         * best_x permanecem fixos durante todo o passo. */
        for (int i = 0; i < n; i++) {
            if (rng_uniform(&rng) < z_current) {
                /* (a) Reinicialização com probabilidade z. Um ÚNICO escalar u
                 * é compartilhado por todas as coordenadas, como na demo
                 * original. Com limites iguais em todas as dimensões, o
                 * vetor resultante tem chaves todas iguais. */
                double u = rng_uniform(&rng);
                for (int j = 0; j < d; j++)
                    x[(size_t)i * d + j] = c->lower + (c->upper - c->lower) * u;
            } else {
                /* p = tanh(|S(i) - DF|): quanto pior o agente em relação ao
                 * melhor histórico, mais perto de 1 (favorece aproximação). */
                double p = tanh(fabs(s[i] - *best_f));
                for (int j = 0; j < d; j++) vb[j] = (2 * rng_uniform(&rng) - 1) * a;
                for (int j = 0; j < d; j++) vc[j] = (2 * rng_uniform(&rng) - 1) * b;

                for (int j = 0; j < d; j++) {
                    double r = rng_uniform(&rng);
                    /* A e B: agentes sorteados com reposição. Podem coincidir
                     * e podem ser o próprio i. Os sorteios ocorrem mesmo no
                     * ramo (c), para manter a ordem do fluxo aleatório. */
                    int A = rng_index(&rng, n);
                    int B = rng_index(&rng, n);
                    size_t k = (size_t)i * d + j;

                    if (r < p) {
                        /* (b) Aproximação do melhor histórico (best_x). */
                        x[k] = best_x[j] + vb[j] * (w[k] * x[(size_t)A * d + j]
                                                    - x[(size_t)B * d + j]);
                    } else {
                        /* (c) Contração em torno da própria posição. */
                        x[k] = vc[j] * x[k];
                    }
                }
            }
        }

        /* ---- 6. Curva de convergência ----
         * Melhor já AVALIADO após N*t avaliações (monótona não crescente). */
        history[t - 1] = *best_f;
    }
    status = 0;

done:
    free(x); free(w); free(s); free(vb); free(vc); free(rank);
    return status;
}


int sma_solve_ex(const SmaConfig *c, Objective f, void *ctx,
                 double *best_x, double *history, double *best_f,
                 const FuzzyParams *fp, double *z_history,
                 double *div_history, double *stag_history) {
    return sma_solve_impl(c, f, ctx, best_x, history, best_f, fp,
                          z_history, div_history, stag_history);
}

int sma_solve(const SmaConfig *c, Objective f, void *ctx,
              double *best_x, double *history, double *best_f) {
    return sma_solve_impl(c, f, ctx, best_x, history, best_f, NULL, NULL, NULL, NULL);
}

int sma_solve_adaptive(const SmaConfig *c, Objective f, void *ctx,
                       double *best_x, double *history, double *best_f,
                       double *z_history) {
    FuzzyParams fp; sma_fuzzy_defaults(&fp);
    return sma_solve_impl(c, f, ctx, best_x, history, best_f, &fp, z_history, NULL, NULL);
}
