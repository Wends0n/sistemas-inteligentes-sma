#ifndef SMA_H
#define SMA_H
#include <stdint.h>
/* Minimização genérica. O callback não pode modificar x. */
typedef double (*Objective)(const double *x, int d, void *context);
typedef struct { int population, iterations, dimension; double lower, upper, z; uint64_t seed; } SmaConfig;
/* O chamador aloca best_x[dimension] e history[iterations].
 * Retorna 0 em sucesso, -1 para parâmetros, memória ou fitness inválidos.
 * history[t-1] é o melhor histórico após N*t avaliações (t começa em 1).
 * Não avalia a movimentação final, como a demonstração original. */
int sma_solve(const SmaConfig *c, Objective f, void *context,
              double *best_x, double *history, double *best_f);
/* Variante adaptativa: ajusta z a cada geração via controlador fuzzy Sugeno.
 * z_history pode ser NULL; caso contrário recebe o z aplicado em cada geração. */
int sma_solve_adaptive(const SmaConfig *c, Objective f, void *context,
                       double *best_x, double *history, double *best_f,
                       double *z_history);
/* Controlador fuzzy Takagi-Sugeno de ordem zero (Semana 5).
 * Entradas: diversidade D e estagnação E, ambas em [0,1].
 * Cada termo linguístico é uma rampa linear {a,b}:
 *   "baixa": pertinência 1 para v <= a, 0 para v >= b (decrescente);
 *   "alta" : pertinência 0 para v <= a, 1 para v >= b (crescente).
 * out[] são os singletons de z das regras R1..R4:
 *   R1 = D baixa & E baixa, R2 = D baixa & E alta,
 *   R3 = D alta  & E baixa, R4 = D alta  & E alta.
 * z = soma(w_i*out_i)/soma(w_i), com w_i = produto das pertinências. */
typedef struct {
    double div_low[2], div_high[2], stag_low[2], stag_high[2], out[4];
    double stag_window;   /* E = gerações sem melhora / max(10, stag_window*T), em [0,1] */
} FuzzyParams;
/* Preenche os parâmetros padrão (calibrados com os trials do Optuna). */
void sma_fuzzy_defaults(FuzzyParams *fp);
/* Avalia o controlador. Retorna 0.04 se nenhuma regra dispara. */
double sma_fuzzy_z(const FuzzyParams *fp, double diversity, double stagnation);
/* Versão completa. fp == NULL executa com z fixo (c->z), mas ainda registra
 * os traços. Qualquer vetor de saída (z_history, div_history, stag_history)
 * pode ser NULL; os não nulos recebem o valor de cada geração. */
int sma_solve_ex(const SmaConfig *c, Objective f, void *context,
                 double *best_x, double *history, double *best_f,
                 const FuzzyParams *fp, double *z_history,
                 double *div_history, double *stag_history);
/* Função exposta para testar separadamente a equação dos pesos. */
double sma_weight(double fitness, double best, double worst, int rank, int n, double u);
#endif
