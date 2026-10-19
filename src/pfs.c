#include "pfs.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
int pfs_generate(Pfs *p,int n,int m,int64_t seed) {
    if (!p || n<1 || n>PFS_MAX_JOBS || m<1 || m>PFS_MAX_MACHINES || seed<1 || seed>=2147483647) return -1;
    memset(p,0,sizeof *p); p->jobs=n; p->machines=m;
    /* Park-Miller, ordem máquina -> tarefa prescrita por Taillard. */
    for (int k=0;k<m;k++) for (int j=0;j<n;j++) {
        seed=(16807*seed)%2147483647;
        p->p[k][j]=1+(int)(99.0*((double)seed/2147483647.0));
    }
    return 0;
}
int pfs_write(const Pfs *p,const char *path) {
    FILE *f=fopen(path,"w"); if (!f) return -1;
    fprintf(f,"%d %d\n",p->jobs,p->machines);
    for (int k=0;k<p->machines;k++) {
        for (int j=0;j<p->jobs;j++) fprintf(f,"%d%c",p->p[k][j],j+1==p->jobs?'\n':' ');
    }
    int bad=ferror(f); return fclose(f)==0 && !bad ? 0 : -1;
}
void pfs_decode(const double *x,int n,int *perm) {
    for (int i=0;i<n;i++) {
        int j=i;
        while (j>0 && x[perm[j-1]]>x[i]) { perm[j]=perm[j-1]; j--; }
        perm[j]=i;
    }
}
int pfs_valid(const int *perm,int n) {
    if (!perm || n<1 || n>PFS_MAX_JOBS) return 0;
    int seen[PFS_MAX_JOBS]={0};
    for (int i=0;i<n;i++) if (perm[i]<0 || perm[i]>=n || seen[perm[i]]++) return 0;
    return 1;
}
int64_t pfs_makespan(const Pfs *p,const int *perm) {
    if (!p || p->machines<1 || p->machines>PFS_MAX_MACHINES || !pfs_valid(perm,p->jobs)) return -1;
    int64_t end[PFS_MAX_MACHINES]={0};
    /* C[i,k] = max(C[i-1,k],C[i,k-1]) + p[k,perm[i]]. */
    for (int i=0;i<p->jobs;i++) {
        int64_t previous=0;
        for (int k=0;k<p->machines;k++) {
            if (p->p[k][perm[i]]<=0) return -1;
            if (end[k]<previous) end[k]=previous;
            end[k]+=p->p[k][perm[i]];
            previous=end[k];
        }
    }
    return end[p->machines-1];
}
double pfs_objective(const double *x,int d,void *context) {
    Pfs *p=context; int perm[PFS_MAX_JOBS];
    if (!p || d!=p->jobs || d<1 || d>PFS_MAX_JOBS) return NAN;
    for (int i=0;i<d;i++) if (!isfinite(x[i])) return NAN;
    pfs_decode(x,d,perm);
    int64_t v=pfs_makespan(p,perm);
    return v<0 ? NAN : (double)v;
}
