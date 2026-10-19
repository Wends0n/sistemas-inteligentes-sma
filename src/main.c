/**
 * @file main.c
 * @brief Experimento: SMA aplicado ao PFS nas instâncias ta001 e ta011.
 *
 * Uso: sma_pfs PASTA [execucoes=30] [iteracoes=500] [populacao=30]
 * A PASTA deve existir. Arquivos gerados nela:
 *   ta001.txt, ta011.txt  matrizes de tempos (máquina x tarefa)
 *   history.csv           melhor makespan por iteração, de cada execução
 *   runs.csv              resultado final de cada execução (+ permutação)
 *
 * Protocolo:
 *   - instâncias Taillard regeneradas de suas sementes oficiais;
 *   - cada execução usa semente própria e estado totalmente novo;
 *   - domínio das chaves [-1,1], z = 0,03 (parâmetros do SMA original).
 * As estatísticas (média, DP) e o gráfico ficam em scripts/analyze.py.
 */
#include "sma.h"
#include "pfs.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

/* Converte texto em inteiro no intervalo [1, max]; devolve -1 se inválido. */
static int positive(const char *s, int max) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    return (errno || *end || end == s || v < 1 || v > max) ? -1 : (int)v;
}

/* Abre "dir/name" para escrita; NULL em erro ou caminho longo demais. */
static FILE *output(const char *dir, const char *name) {
    char path[4096];
    if (snprintf(path, sizeof path, "%s/%s", dir, name) >= (int)sizeof path)
        return NULL;
    return fopen(path, "w");
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr, "Uso: %s pasta_resultados [execucoes=30] "
                "[iteracoes=500] [populacao=30]\nA pasta deve existir.\n", argv[0]);
        return 1;
    }
    int runs = argc > 2 ? positive(argv[2], 10000)   : 30;
    int it   = argc > 3 ? positive(argv[3], 1000000) : 500;
    int pop  = argc > 4 ? positive(argv[4], 10000)   : 30;
    if (runs < 1 || it < 1 || pop < 1) {
        fprintf(stderr, "Parametro inteiro invalido\n");
        return 1;
    }

    FILE *hist  = output(argv[1], "history.csv");
    FILE *final = output(argv[1], "runs.csv");
    if (!hist || !final) {
        if (hist)  fclose(hist);
        if (final) fclose(final);
        perror("saida");
        return 1;
    }
    fprintf(hist,  "instance,run,seed,iteration,evaluations,best_makespan\n");
    fprintf(final, "instance,run,seed,population,iterations,evaluations,"
                   "best_makespan,cpu_seconds,permutation\n");

    double *curve = malloc((size_t)it * sizeof *curve);   /* histórico de uma execução */
    if (!curve) { fclose(hist); fclose(final); return 1; }

    /* Instâncias de Taillard: 20 tarefas; sementes oficiais dos tempos. */
    const char   *names[]    = { "ta001", "ta011" };
    const int     machines[] = { 5, 10 };
    const int64_t seeds[]    = { 873654221, 587595453 };

    int status = 1;   /* 0 somente se tudo terminar sem erro */
    for (int b = 0; b < 2; b++) {
        Pfs p;
        double x[20], best;
        int perm[20];
        char path[4096];

        /* Gera os tempos e grava a matriz para conferência externa. */
        if (pfs_generate(&p, 20, machines[b], seeds[b])) goto done;
        if (snprintf(path, sizeof path, "%s/%s.txt", argv[1], names[b])
                >= (int)sizeof path || pfs_write(&p, path)) goto done;

        for (int r = 1; r <= runs; r++) {
            /* Semente do solver distinta por execução e por instância
             * (e diferente da semente do problema), para reprodutibilidade. */
            uint64_t seed = UINT64_C(202610190000) + (uint64_t)b * 100000 + (uint64_t)r;
            SmaConfig cfg = { pop, it, 20, -1, 1, 0.03, seed };

            clock_t start = clock();
            if (sma_solve(&cfg, pfs_objective, &p, x, curve, &best)) goto done;
            double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;

            /* Autoverificação: a solução devolvida deve decodificar para uma
             * permutação válida cujo makespan seja exatamente `best`. */
            pfs_decode(x, 20, perm);
            if (!pfs_valid(perm, 20) || (double)pfs_makespan(&p, perm) != best)
                goto done;

            /* Curva da execução; deve ser monótona não crescente. */
            for (int t = 1; t <= it; t++) {
                if (t > 1 && curve[t - 1] > curve[t - 2]) goto done;
                fprintf(hist, "%s,%d,%" PRIu64 ",%d,%" PRId64 ",%.0f\n",
                        names[b], r, seed, t, (int64_t)pop * t, curve[t - 1]);
            }

            /* Resultado final; permutação em numeração externa (1..20). */
            fprintf(final, "%s,%d,%" PRIu64 ",%d,%d,%" PRId64 ",%.0f,%.6f,\"",
                    names[b], r, seed, pop, it, (int64_t)pop * it, best, seconds);
            for (int j = 0; j < 20; j++)
                fprintf(final, "%d%s", perm[j] + 1, j == 19 ? "\"\n" : " ");
        }
        printf("%s: %d execucoes concluidas\n", names[b], runs);
    }
    status = (ferror(hist) || ferror(final)) ? 1 : 0;

done:
    free(curve);
    if (fclose(hist))  status = 1;
    if (fclose(final)) status = 1;
    if (status) fprintf(stderr, "Falha no experimento; resultados podem estar incompletos.\n");
    return status;
}
