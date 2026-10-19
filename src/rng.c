#include "rng.h"
uint64_t rng_next(Rng *r) {
    uint64_t z = (r->state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
double rng_uniform(Rng *r) { return (rng_next(r) >> 11) * 0x1.0p-53; }
int rng_index(Rng *r, int n) {
    uint64_t bound = (uint64_t)n, threshold = (uint64_t)(-bound) % bound;
    uint64_t v;
    do { v = rng_next(r); } while (v < threshold);
    return (int)(v % bound);
}
