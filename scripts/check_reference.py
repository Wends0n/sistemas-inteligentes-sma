"""Oráculo de teste pequeno das equações do SMA; não é usado no experimento.
Usa o mesmo fluxo pseudoaleatório para localizar erros de tradução C/Python.
Não implica equivalência bit a bit com o RNG do MATLAB.
"""
import csv
import math
from pathlib import Path
MASK=(1<<64)-1
class Rng:
    def __init__(self,seed): self.state=seed
    def next(self):
        self.state=(self.state+0x9e3779b97f4a7c15)&MASK
        z=self.state
        z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK
        z=((z^(z>>27))*0x94d049bb133111eb)&MASK
        return z^(z>>31)
    def uniform(self): return (self.next()>>11)*2**-53
    def index(self,n):
        threshold=((1<<64)-n)%n
        while True:
            u=self.next()
            if u>=threshold: return u%n

def reference(z):
    rng=Rng(123); n=6; d=4; T=12
    x=[[2*rng.uniform()-1 for _ in range(d)] for _ in range(n)]
    best=math.inf; trace=[]
    for t in range(1,T+1):
        x=[[min(1,max(-1,v)) for v in row] for row in x]
        s=[sum(v*v for v in row) for row in x]
        order=sorted(range(n),key=lambda i:s[i]); lo=s[order[0]];hi=s[order[-1]]
        w=[[0.]*d for _ in range(n)]
        for rank,i in enumerate(order):
            for j in range(d):
                q=(s[i]-lo)/(hi-lo) if hi!=lo else 0
                w[i][j]=1+(1 if rank<n//2 else -1)*rng.uniform()*math.log10(1+q)
        if lo<best: best=lo; xb=x[order[0]].copy()
        b=1-t/T;a=math.atanh(b)
        for i in range(n):
            if rng.uniform()<z:
                u=rng.uniform();x[i]=[2*u-1]*d
            else:
                p=math.tanh(abs(s[i]-best))
                vb=[(2*rng.uniform()-1)*a for _ in range(d)]
                vc=[(2*rng.uniform()-1)*b for _ in range(d)]
                for j in range(d):
                    r=rng.uniform();A=rng.index(n);B=rng.index(n)
                    x[i][j]=xb[j]+vb[j]*(w[i][j]*x[A][j]-x[B][j]) if r<p else vc[j]*x[i][j]
        trace.append(best)
    return trace
rows=list(csv.DictReader(Path('build/parity.csv').open()))
for case,z in enumerate((0,.03,1)):
    actual=[float(r['best']) for r in rows if int(r['case'])==case]
    expected=reference(z)
    assert len(actual)==12
    for a,b in zip(actual,expected): assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-15),(case,a,b)
print('OK: 36 pontos de histórico C conferem com oráculo independente das equações.')
