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
/* Função exposta para testar separadamente a equação dos pesos. */
double sma_weight(double fitness, double best, double worst, int rank, int n, double u);
#endif
