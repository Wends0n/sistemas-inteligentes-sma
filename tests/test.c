#include "sma.h"
#include "pfs.h"
#include "rng.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int calls;
static double sphere(const double *x,int d,void *ctx) {
    (void)ctx; double s=0; calls++;
    for(int j=0;j<d;j++) { assert(isfinite(x[j]) && x[j]>=-1 && x[j]<=1); s+=x[j]*x[j]; }
    return s;
}
static double constant(const double *x,int d,void *ctx) { (void)x;(void)d;(void)ctx;return 7; }
static double invalid(const double *x,int d,void *ctx) { (void)x;(void)d;(void)ctx;return NAN; }
/* Oráculo independente: tabela 2D, diferente do vetor de conclusão usado no solver. */
static int64_t oracle(const Pfs *p,const int *perm) {
    int64_t c[7][4]={{0}};
    for(int i=1;i<=p->jobs;i++) for(int k=1;k<=p->machines;k++)
        c[i][k]=(c[i-1][k]>c[i][k-1]?c[i-1][k]:c[i][k-1])+p->p[k-1][perm[i-1]];
    return c[p->jobs][p->machines];
}
static int permutations;
static int64_t exact=INT64_MAX;
static void enumerate(const Pfs *p,int *perm,int pos) {
    if(pos==p->jobs) {
        int64_t a=pfs_makespan(p,perm),b=oracle(p,perm);
        assert(a==b); if(a<exact)exact=a; permutations++; return;
    }
    for(int j=pos;j<p->jobs;j++) {
        int v=perm[pos];perm[pos]=perm[j];perm[j]=v;
        enumerate(p,perm,pos+1);
        v=perm[pos];perm[pos]=perm[j];perm[j]=v;
    }
}
int main(void) {
    Rng rng={0}; assert(rng_next(&rng)==UINT64_C(0xe220a8397b1dcdaf));
    for(int i=0;i<10000;i++) { double u=rng_uniform(&rng);assert(u>=0 && u<1);int k=rng_index(&rng,7);assert(k>=0 && k<7); }
    Pfs p;
    assert(pfs_generate(&p,20,5,873654221)==0);
    const int official[5][20]={
        {54,83,15,71,77,36,53,38,27,87,76,91,14,29,12,77,32,87,68,94},
        {79,3,11,99,56,70,99,60,5,56,3,61,73,75,47,14,21,86,5,77},
        {16,89,49,15,89,45,60,23,57,64,7,1,63,41,63,47,26,75,77,40},
        {66,58,31,68,78,91,13,59,49,85,85,9,39,41,56,40,54,77,51,31},
        {58,56,20,85,53,35,53,41,69,13,86,72,8,49,47,87,58,18,68,28}};
    for(int k=0;k<5;k++)for(int j=0;j<20;j++)assert(p.p[k][j]==official[k][j]);
    assert(pfs_generate(&p,0,5,1)==-1);assert(pfs_generate(&p,20,5,0)==-1);
    double keys[]={0.5,-0.5,0.5,0.1}; int perm[6];
    pfs_decode(keys,4,perm);assert(perm[0]==1 && perm[1]==3 && perm[2]==0 && perm[3]==2);
    int duplicate[]={0,0};assert(!pfs_valid(duplicate,2));
    assert(pfs_generate(&p,6,3,123456)==0);
    for(int i=0;i<6;i++)perm[i]=i;
    enumerate(&p,perm,0);assert(permutations==720);
    double x[6],h[100],x2[6],h2[100],best,best2;
    SmaConfig c={10,100,6,-1,1,0.03,42};
    assert(sma_solve(&c,pfs_objective,&p,x,h,&best)==0);assert(best>=exact);
    assert(sma_solve(&c,pfs_objective,&p,x2,h2,&best2)==0);
    assert(best==best2 && !memcmp(x,x2,sizeof x) && !memcmp(h,h2,sizeof h));
    pfs_decode(x,6,perm);assert(pfs_makespan(&p,perm)==best);
    printf("Exaustao: 720 permutacoes conferidas; otimo pequeno=%lld, SMA=%.0f\n",(long long)exact,best);
    calls=0; assert(sma_solve(&c,sphere,NULL,x,h,&best)==0);assert(calls==1000);
    for(int t=1;t<100;t++)assert(h[t]<=h[t-1]);
    assert(fabs(sphere(x,6,NULL)-best)<1e-15);
    assert(sma_solve(&c,constant,NULL,x,h,&best)==0 && best==7);
    assert(sma_weight(7,7,7,0,30,0.5)==1);
    assert(fabs(sma_weight(10,0,10,29,30,1)-(1-log10(2)))<1e-15);
    assert(fabs(sma_weight(5,0,10,0,30,1)-(1+log10(1.5)))<1e-15);
    assert(sma_solve(&c,invalid,NULL,x,h,&best)==-1);
    c.iterations=1;c.population=1;
    calls=0;assert(sma_solve(&c,sphere,NULL,x,h,&best)==0 && calls==1);
    c.z=1;assert(sma_solve(&c,sphere,NULL,x,h,&best)==0);
    c.z=NAN;assert(sma_solve(&c,sphere,NULL,x,h,&best)==-1);
    c.z=0;c.lower=c.upper;assert(sma_solve(&c,sphere,NULL,x,h,&best)==-1);
    /* Traços pequenos para comparação independente em Python. */
    FILE *f=fopen("build/parity.csv","w");assert(f);
    fprintf(f,"case,iteration,best\n");
    for(int k=0;k<3;k++) {
        c=(SmaConfig){6,12,4,-1,1,k==0?0:k==1?0.03:1,123};
        assert(sma_solve(&c,sphere,NULL,x,h,&best)==0);
        for(int t=0;t<12;t++)fprintf(f,"%d,%d,%.17g\n",k,t+1,h[t]);
    }
    assert(fclose(f)==0);
    puts("OK: RNG, Taillard, SPV, makespan, pesos, limites, empate, contagem, reproducibilidade e parametros.");
    return 0;
}
