#ifndef RNG_H
#define RNG_H
#include <stdint.h>
/* SplitMix64: estado local a cada execução; nenhuma chamada a rand(). */
typedef struct { uint64_t state; } Rng;
uint64_t rng_next(Rng *r);
double rng_uniform(Rng *r); /* [0,1), 53 bits */
int rng_index(Rng *r, int n); /* uniforme em [0,n), por rejeição */
#endif
