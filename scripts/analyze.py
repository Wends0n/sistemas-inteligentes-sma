#!/usr/bin/env python3
"""Valida arquivos brutos, agrega 30 execuções e desenha média +/- DP amostral.
Python é usado apenas depois do solver C: validação, estatística e gráficos.
NEH é uma comparação determinística separada, nunca inserida no SMA.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def makespan(p,perm):
    # Tabela 2D independente da avaliação C (que usa memória O(m)).
    c=np.zeros((len(perm)+1,p.shape[0]+1),dtype=np.int64)
    for i,job in enumerate(perm,1):
        for k in range(1,p.shape[0]+1):
            c[i,k]=max(c[i-1,k],c[i,k-1])+p[k-1,job]
    return int(c[-1,-1])


def neh(p):
    order=sorted(range(p.shape[1]),key=lambda j:(-int(p[:,j].sum()),j))
    sequence=[]
    for job in order:
        candidates=[sequence[:i]+[job]+sequence[i:] for i in range(len(sequence)+1)]
        # Empate: primeira posição de inserção. Sem melhorias locais extras.
        sequence=min(candidates,key=lambda s:makespan(p,s))
    return sequence


def main():
    out=Path(sys.argv[1] if len(sys.argv)>1 else 'results')
    with (out/'history.csv').open() as f: rows=list(csv.DictReader(f))
    with (out/'runs.csv').open() as f: runs=list(csv.DictReader(f))
    names=sorted({r['instance'] for r in runs})
    fig,axes=plt.subplots(1,len(names),figsize=(12.8,5.6),squeeze=False)
    summary=[]; aggregation=[]
    for ax,name in zip(axes[0],names):
        records=[r for r in runs if r['instance']==name]
        n,m=map(int,(out/f'{name}.txt').read_text().splitlines()[0].split())
        p=np.loadtxt(out/f'{name}.txt',skiprows=1,dtype=np.int64)
        assert p.shape==(m,n)
        curves=[];seeds=set();iterations={int(r['iterations']) for r in records}
        assert len(iterations)==1 and len(records)>=2
        T=iterations.pop()
        for r in records:
            seed=int(r['seed']);assert seed not in seeds;seeds.add(seed)
            h=[v for v in rows if v['instance']==name and v['run']==r['run']]
            assert len(h)==T and [int(v['iteration']) for v in h]==list(range(1,T+1))
            assert all(int(v['seed'])==seed and int(v['evaluations'])==int(r['population'])*int(v['iteration']) for v in h)
            c=np.array([float(v['best_makespan']) for v in h])
            assert np.isfinite(c).all() and np.all(np.diff(c)<=0)
            perm=[int(j)-1 for j in r['permutation'].split()]
            assert sorted(perm)==list(range(n))
            assert makespan(p,perm)==int(r['best_makespan'])==c[-1]
            assert int(r['evaluations'])==int(r['population'])*T
            curves.append(c)
        a=np.stack(curves);mean=a.mean(axis=0);sd=a.std(axis=0,ddof=1)
        baseline_perm=neh(p);baseline=makespan(p,baseline_perm)
        lower=int(max(p.sum(axis=0).max(),p.sum(axis=1).max()))
        assert (a>=lower).all()
        for t in range(T):aggregation.append([name,t+1,len(a),mean[t],sd[t]])
        result=dict(instance=name,jobs=n,machines=m,runs=len(a),population=int(records[0]['population']),
                    iterations=T,evaluations_per_run=int(records[0]['evaluations']),
                    best=int(a[:,-1].min()),mean=float(mean[-1]),sample_sd=float(sd[-1]),
                    median=float(np.median(a[:,-1])),worst=int(a[:,-1].max()),
                    initial_mean=float(mean[0]),improvement_percent=float(100*(mean[0]-mean[-1])/mean[0]),
                    neh=baseline,neh_permutation=[j+1 for j in baseline_perm],
                    mean_gap_to_neh_percent=float(100*(mean[-1]-baseline)/baseline),
                    simple_lower_bound=lower,mean_cpu_seconds=float(np.mean([float(r['cpu_seconds']) for r in records])))
        summary.append(result)
        t=np.arange(1,T+1)
        ax.fill_between(t,mean-sd,mean+sd,color='#4479a9',alpha=.19,label='Média ± 1 DP amostral')
        ax.plot(t,mean,color='#235d88',lw=2,label='Média do melhor histórico')
        ax.axhline(baseline,color='#b77623',ls='--',lw=1.4,label=f'NEH separada: {baseline}')
        ax.set(title=f'{name} · {n} tarefas × {m} máquinas',xlabel='Iteração',ylabel='Aptidão: makespan (menor é melhor)')
        ax.set_xlim(1,T);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper right',fontsize=8.5)
        ax.text(.98,.53,f'Final: {mean[-1]:.2f} ± {sd[-1]:.2f}',transform=ax.transAxes,ha='right',fontsize=10,color='#235d88')
    fig.suptitle('SMA com chaves reais · convergência no Flowshop Permutacional',fontsize=15,fontweight='bold',y=.97)
    fig.text(.5,.89,f'{len(records)} execuções por instância · N = {records[0]["population"]} · T = {T} · z = 0,03 · domínio [−1, 1]',ha='center',fontsize=10)
    fig.text(.5,.025,'Faixa = dispersão entre execuções (não é intervalo de confiança). NEH é referência viável, não certificado de ótimo.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(top=.80,bottom=.16,wspace=.24,left=.07,right=.98)
    fig.savefig(out/'convergencia.png',dpi=190)
    fig.savefig(out/'convergencia.svg')
    with (out/'convergence_stats.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['instance','iteration','runs','mean','sample_sd']);w.writerows(aggregation)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    fields=[k for k in summary[0] if k!='neh_permutation']
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print(f'OK: {len(runs)} soluções, {len(rows)} pontos e estatísticas validados.')
if __name__=='__main__':main()
