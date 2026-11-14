#!/usr/bin/env python3
"""Semana 4: LHS + MLP surrogate (métricas por instância, tempo real vs surrogate).
Semana 5 (integração): SMA adaptativo (fuzzy) com o surrogate no loop.
O controlador fuzzy é calibrado/avaliado em scripts/fuzzy_optuna.py."""
from __future__ import annotations
import argparse, csv, json, math, time, ctypes as ct
import numpy as np
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from smalib import (ROOT, OUT, FuzzyParams, build_lib, decode, instance, makespan, objectives, run_solver)

SUR, FUZZY = OUT / "surrogate", OUT / "fuzzy"
for d in (SUR, FUZZY): d.mkdir(parents=True, exist_ok=True)

def fit_mlp(Xtr, ytr, seed, hidden=(32, 16)):
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(ytr.reshape(-1, 1))
    m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=1e-4,
                     learning_rate_init=.002, max_iter=1400, early_stopping=True,
                     validation_fraction=.15, n_iter_no_change=50, random_state=seed)
    m.fit(sx.transform(Xtr), sy.transform(ytr.reshape(-1, 1)).ravel())
    return m, sx, sy

def predict(m, sx, sy, X):
    return sy.inverse_transform(m.predict(sx.transform(X)).reshape(-1, 1)).ravel()

def reg_metrics(y, p):
    mse = float(mean_squared_error(y, p))
    return {"mse": mse, "rmse": math.sqrt(mse), "r2": float(r2_score(y, p))}

def export_c_model(models, scalers, target_scalers):
    lines = ['#include <math.h>', '#include <stddef.h>', '#include "pfs.h"']
    for k,(model,scaler,ys) in enumerate(zip(models,scalers,target_scalers)):
        lines.append(f'static const double MU{k}[20] = '+'{'+','.join(f'{v:.17g}' for v in scaler.mean_)+'};')
        lines.append(f'static const double SCALE{k}[20] = '+'{'+','.join(f'{v:.17g}' for v in scaler.scale_)+'};')
        for idx, arr in enumerate(model.coefs_):
            flat=arr.flatten(order='C')
            lines.append(f'static const double W{k}_{idx}[{arr.size}] = '+'{'+','.join(f'{v:.17g}' for v in flat)+'};')
        for idx, arr in enumerate(model.intercepts_):
            lines.append(f'static const double B{k}_{idx}[{arr.size}] = '+'{'+','.join(f'{v:.17g}' for v in arr)+'};')
        lines.append(f'static const double YMEAN{k}={float(ys.mean_[0]):.17g}, YSCALE{k}={float(ys.scale_[0]):.17g};')
    lines += [
      'static double predict_one(const double*x,int which) {',
      ' double a[32],b[16],z[20],out=0;',
      ' if(which==0){',
      ' for(int i=0;i<20;i++)z[i]=(x[i]-MU0[i])/SCALE0[i];',
      ' for(int j=0;j<32;j++){double v=B0_0[j];for(int i=0;i<20;i++)v+=z[i]*W0_0[i*32+j];a[j]=v>0?v:0;}',
      ' for(int j=0;j<16;j++){double v=B0_1[j];for(int i=0;i<32;i++)v+=a[i]*W0_1[i*16+j];b[j]=v>0?v:0;}',
      ' out=B0_2[0];for(int i=0;i<16;i++)out+=b[i]*W0_2[i];return out*YSCALE0+YMEAN0;',
      ' } else {',
      ' for(int i=0;i<20;i++)z[i]=(x[i]-MU1[i])/SCALE1[i];',
      ' for(int j=0;j<32;j++){double v=B1_0[j];for(int i=0;i<20;i++)v+=z[i]*W1_0[i*32+j];a[j]=v>0?v:0;}',
      ' for(int j=0;j<16;j++){double v=B1_1[j];for(int i=0;i<32;i++)v+=a[i]*W1_1[i*16+j];b[j]=v>0?v:0;}',
      ' out=B1_2[0];for(int i=0;i<16;i++)out+=b[i]*W1_2[i];return out*YSCALE1+YMEAN1;',
      ' }}',
      'double surrogate_objective(const double*x,int d,void*ctx){',
      ' if(!x||d!=20||!ctx)return NAN;',
      ' const Pfs*p=(const Pfs*)ctx; if(p->machines==5)return predict_one(x,0);',
      ' if(p->machines==10)return predict_one(x,1); return NAN;',
      '}'
    ]
    (ROOT/'src'/'surrogate_model.c').write_text('\n'.join(lines)+'\n')

def timed_runs(lib, p, cb, reps, pop, iters, seed0):
    out = []
    for rep in range(reps):
        r = run_solver(lib, p, cb, pop, iters, seed0 + rep)
        out.append((r["time"], r["best"], makespan(p, decode(r["x"]))))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--costs", type=int, nargs="+", default=[100, 1000, 5000],
                    help="repetições do makespan para simular um objetivo caro")
    ap.add_argument("--learning-curve", action="store_true")
    args = ap.parse_args()
    if args.samples < 200: ap.error("--samples deve ser >= 200")
    names = ("ta001", "ta011")
    insts = {"ta001": instance(20, 5, 873654221), "ta011": instance(20, 10, 587595453)}
    data, models, test, train_time = {}, [], {}, {}
    per_inst, preds, truths = {}, [], []
    for idx, (name, p) in enumerate(insts.items()):
        sampler = qmc.LatinHypercube(d=20, seed=args.seed + idx)
        X = qmc.scale(sampler.random(args.samples), -1, 1)
        y = np.array([makespan(p, decode(r)) for r in X], dtype=float)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=args.seed + idx)
        t0 = time.perf_counter(); m, sx, sy = fit_mlp(Xtr, ytr, args.seed + idx); train_time[name] = time.perf_counter() - t0
        pr = predict(m, sx, sy, Xte)
        data[name] = (X, y); models.append((m, sx, sy)); test[name] = (yte, pr)
        per_inst[name] = {**reg_metrics(yte, pr), "n_treino": len(ytr), "n_teste": len(yte),
                          "media_y_teste": float(yte.mean()), "desvio_y_teste": float(yte.std()),
                          "iteracoes_treino": int(m.n_iter_), "tempo_treino_s": train_time[name]}
        preds += pr.tolist(); truths += yte.tolist()
    pooled = reg_metrics(truths, preds)
    metrics = {"model": "MLPRegressor (20-32-16-1), ReLU/linear; um modelo por instância",
               "sampling": "Latin Hypercube Sampling", "samples_per_instance": args.samples,
               "total_samples": 2 * args.samples, "features": 20,
               "por_instancia": per_inst,
               "mse": float(np.mean([v["mse"] for v in per_inst.values()])),
               "rmse": float(np.mean([v["rmse"] for v in per_inst.values()])),
               "r2": float(np.mean([v["r2"] for v in per_inst.values()])),
               "nota_metricas_agregadas": "mse/rmse/r2 acima = média das métricas por instância (métrica oficial).",
               "agregado_empilhado_nao_usar_para_r2": {**pooled,
                   "aviso": "R2 calculado sobre as duas instâncias empilhadas é inflado pela diferença entre as médias de ta001 e ta011 e NÃO mede a qualidade do surrogate; mantido só para comparação com a versão anterior."}}
    (SUR / "metricas_rna.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (SUR / "dataset_lhs.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["instancia"] + [f"x{i+1}" for i in range(20)] + ["makespan_real"])
        for name in names:
            for row, val in zip(*data[name]): w.writerow([name, *row, val])
    export_c_model([m for m, _, _ in models], [s for _, s, _ in models], [s for _, _, s in models])
    (SUR / "modelo_info.json").write_text(json.dumps({"arquitetura": [20, 32, 16, 1],
        "modelos": {"ta001": "modelo 0 (5 máquinas)", "ta011": "modelo 1 (10 máquinas)"},
        "ativacao_ocultas": "ReLU", "saida": "linear", "normalizacao": "StandardScaler para X e y",
        "pesos_exportados_para": "src/surrogate_model.c"}, indent=2, ensure_ascii=False))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axs, names):
        yt, yp = test[name]; ax.scatter(yt, yp, s=18, alpha=.65)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max()); ax.plot([lo, hi], [lo, hi], "--", c="gray")
        ax.set_title(f"{name}: R²={per_inst[name]['r2']:.3f}  RMSE={per_inst[name]['rmse']:.1f}")
        ax.set_xlabel("Makespan real"); ax.set_ylabel("Predição MLP"); ax.grid(alpha=.3)
    plt.suptitle("Surrogate MLP — real vs. predito (conjunto de teste, por instância)")
    plt.tight_layout(); plt.savefig(SUR / "regressao_real_vs_predito.png", dpi=160); plt.close()

    lib = build_lib(); ob = objectives(lib)
    pop, iters = args.population, args.iterations; evals = pop * iters
    base, rows_cost = [], []
    for name, p in insts.items():
        real = timed_runs(lib, p, ob["real"], 5, pop, iters, args.seed + 1)
        sur = timed_runs(lib, p, ob["surrogate"], 5, pop, iters, args.seed + 1)
        tr, ts = float(np.median([v[0] for v in real])), float(np.median([v[0] for v in sur]))
        base.append({"instancia": name, "repeticoes_tempo": 5, "avaliacoes_por_execucao": evals,
            "tempo_mediano_objetivo_real_s": tr, "tempo_mediano_surrogate_s": ts,
            "us_por_avaliacao_real": 1e6 * tr / evals, "us_por_avaliacao_surrogate": 1e6 * ts / evals,
            "ganho_tempo_percentual": 100 * (tr - ts) / tr,
            "makespan_medio_solucao_real": float(np.mean([v[2] for v in real])),
            "makespan_medio_solucao_surrogate_reavaliado_real": float(np.mean([v[2] for v in sur])),
            "makespan_medio_previsto_surrogate": float(np.mean([v[1] for v in sur]))})
        rows_cost.append({"instancia": name, "custo_objetivo": "real nativo (1x)", "repeticoes_makespan": 1,
            "tempo_mediano_real_s": tr, "tempo_mediano_surrogate_s": ts, "speedup": tr / ts,
            "ganho_tempo_percentual": 100 * (tr - ts) / tr, "tempo_construcao_surrogate_s": float("nan"), "execucoes_para_empatar": float("nan")})
        for R in args.costs:
            lib.pfs_set_cost_repeat(R)
            reps = 3 if R >= 1000 else 5
            rc = timed_runs(lib, p, ob["real_caro"], reps, pop, iters, args.seed + 1)
            tc = float(np.median([v[0] for v in rc])); per_eval = tc / evals
            build = args.samples * per_eval + train_time[name]   # coletar LHS com o objetivo caro + treinar
            gain = tc - ts
            rows_cost.append({"instancia": name, "custo_objetivo": f"simulado {R}x ({1e3*per_eval:.3f} ms/aval.)", "repeticoes_makespan": R,
                "tempo_mediano_real_s": tc, "tempo_mediano_surrogate_s": ts, "speedup": tc / ts,
                "ganho_tempo_percentual": 100 * gain / tc, "tempo_construcao_surrogate_s": build,
                "execucoes_para_empatar": build / gain if gain > 0 else float("inf")})
        lib.pfs_set_cost_repeat(1)
    with (SUR / "comparacao_tempo.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base[0].keys()); w.writeheader(); w.writerows(base)
    with (SUR / "comparacao_tempo_custo.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_cost[0].keys()); w.writeheader(); w.writerows(rows_cost)

    # SMA adaptativo (fuzzy) com o surrogate no loop, uma execução por instância.
    fp = FuzzyParams(); lib.sma_fuzzy_defaults(ct.byref(fp))
    hist_rows = []
    for idx, (name, p) in enumerate(insts.items()):
        r = run_solver(lib, p, ob["surrogate"], pop, iters, args.seed + 20 + idx, 0.03, fp)
        for g in range(iters):
            hist_rows.append({"instancia": name, "geracao": g + 1, "melhor_fitness_previsto": r["hist"][g], "z": r["z"][g],
                              "diversidade": r["diversity"][g], "estagnacao": r["stagnation"][g]})
    with (FUZZY / "historico_controlador_surrogate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist_rows[0].keys()); w.writeheader(); w.writerows(hist_rows)
    plt.figure(figsize=(9, 5))
    for name in names:
        rr = [r for r in hist_rows if r["instancia"] == name]
        plt.plot([r["geracao"] for r in rr], [r["z"] for r in rr], label=name)
    plt.xlabel("Geração"); plt.ylabel("z (taxa de reinicialização)"); plt.title("Fuzzy adaptativo com surrogate no loop — variação de z")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FUZZY / "variacao_z_surrogate.png", dpi=160); plt.close()

    if args.learning_curve:
        lc = []
        for idx, (name, p) in enumerate(insts.items()):
            X = qmc.scale(qmc.LatinHypercube(d=20, seed=args.seed + 50 + idx).random(6000), -1, 1)
            y = np.array([makespan(p, decode(r)) for r in X], dtype=float)
            for n in (500, 1500, 3000, 6000):
                Xtr, Xte, ytr, yte = train_test_split(X[:n], y[:n], test_size=.2, random_state=args.seed + idx)
                m, sx, sy = fit_mlp(Xtr, ytr, args.seed + idx)
                lc.append({"instancia": name, "amostras": n, **reg_metrics(yte, predict(m, sx, sy, Xte))})
        with (SUR / "curva_aprendizado.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=lc[0].keys()); w.writeheader(); w.writerows(lc)

    fuzzy_json = FUZZY / "resumo_fuzzy.json"
    summary = {"metrics": metrics, "comparison": base, "comparison_cost": rows_cost,
               "fuzzy": json.loads(fuzzy_json.read_text()) if fuzzy_json.exists() else "execute scripts/fuzzy_optuna.py"}
    (OUT / "semanas_4_5_resumo.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"metrics": metrics, "comparison": base, "comparison_cost": rows_cost}, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
