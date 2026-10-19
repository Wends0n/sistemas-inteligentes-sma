#ifndef PFS_H
#define PFS_H
#include <stdint.h>
#define PFS_MAX_JOBS 500
#define PFS_MAX_MACHINES 20
/* Matriz p[máquina][tarefa]; índices internos começam em zero. */
typedef struct { int jobs, machines; int p[PFS_MAX_MACHINES][PFS_MAX_JOBS]; } Pfs;
int pfs_generate(Pfs *p, int jobs, int machines, int64_t seed);
int pfs_write(const Pfs *p, const char *path);
/* Smallest Position Value (SPV); empates por ID crescente, sem aleatoriedade. */
void pfs_decode(const double *keys, int n, int *permutation);
int pfs_valid(const int *permutation, int n);
/* -1 se permutação ou dimensão inválida; tempos das instâncias devem ser positivos. */
int64_t pfs_makespan(const Pfs *p, const int *permutation);
double pfs_objective(const double *keys, int dimension, void *context);
#endif
