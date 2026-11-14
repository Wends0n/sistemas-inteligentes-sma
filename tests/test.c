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
    c=(SmaConfig){10,100,6,-1,1,0.03,42};
    double zh[100],zh2[100],ha[100],ha2[100],xa[6],xa2[6],ba,ba2;
    assert(sma_solve_adaptive(&c,sphere,NULL,xa,ha,&ba,zh)==0);
    assert(sma_solve_adaptive(&c,sphere,NULL,xa2,ha2,&ba2,zh2)==0);
    assert(ba==ba2 && !memcmp(zh,zh2,sizeof zh));
    for(int t=0;t<100;t++) assert(zh[t]>=0.012 && zh[t]<=0.085);
    {   /* Controlador: limites, regras nos cantos e parâmetros externos. */
        FuzzyParams fp; sma_fuzzy_defaults(&fp);
        assert(fabs(sma_fuzzy_z(&fp,0.0,0.0)-fp.out[0])<1e-12);   /* R1 */
        assert(fabs(sma_fuzzy_z(&fp,0.0,1.0)-fp.out[1])<1e-12);   /* R2 */
        assert(fabs(sma_fuzzy_z(&fp,1.0,0.0)-fp.out[2])<1e-12);   /* R3 */
        assert(fabs(sma_fuzzy_z(&fp,1.0,1.0)-fp.out[3])<1e-12);   /* R4 */
        for(int i=0;i<=20;i++) for(int j=0;j<=20;j++) {
            double zz=sma_fuzzy_z(&fp,i/20.0,j/20.0);
            assert(zz>=fp.out[2]-1e-12 && zz<=fp.out[1]+1e-12);
        }
        /* fp == NULL: z fixo; deve reproduzir sma_solve e registrar traços. */
        double zt[100],dt[100],st[100],xb[6],hb[100],bb;
        assert(sma_solve_ex(&c,sphere,NULL,xb,hb,&bb,NULL,zt,dt,st)==0);
        double xs[6],hs[100],bs; assert(sma_solve(&c,sphere,NULL,xs,hs,&bs)==0);
        assert(bb==bs && !memcmp(hb,hs,sizeof hb));
        for(int t=0;t<100;t++) assert(zt[t]==c.z && dt[t]>=0 && dt[t]<=1 && st[t]>=0 && st[t]<=1);
        /* sma_solve_adaptive == sma_solve_ex com os padrões. */
        double zx[100],dx[100],sx[100];
        assert(sma_solve_ex(&c,sphere,NULL,xb,hb,&bb,&fp,zx,dx,sx)==0);
        assert(bb==ba && !memcmp(zx,zh,sizeof zx));
        /* Rastro coerente: z da geração t = controlador(D_t, E_t). */
        for(int t=0;t<100;t++) assert(fabs(zx[t]-sma_fuzzy_z(&fp,dx[t],sx[t]))<1e-15);
    }
    {   /* Objetivo caro simulado devolve o mesmo valor do real. */
        Pfs q; double key[20]; assert(pfs_generate(&q,20,5,873654221)==0);
        for(int i=0;i<20;i++) key[i]=sin(i*1.7);
        double r0=pfs_objective(key,20,&q); pfs_set_cost_repeat(50);
        assert(pfs_objective_costly(key,20,&q)==r0); pfs_set_cost_repeat(1);
    }
    /* Traços pequenos para comparação independente em Python. */
    FILE *f=fopen("build/parity.csv","w");assert(f);
    fprintf(f,"case,iteration,best\n");
    for(int k=0;k<3;k++) {
        c=(SmaConfig){6,12,4,-1,1,k==0?0:k==1?0.03:1,123};
        assert(sma_solve(&c,sphere,NULL,x,h,&best)==0);
        for(int t=0;t<12;t++)fprintf(f,"%d,%d,%.17g\n",k,t+1,h[t]);
    }
    assert(fclose(f)==0);
    puts("OK: RNG, Taillard, SPV, makespan, pesos, limites, empate, contagem, reproducibilidade, fuzzy adaptativo e parametros.");
    return 0;
}
