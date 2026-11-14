#!/usr/bin/env python3
"""Semana 5: calibra o controlador fuzzy Sugeno com os dados do Optuna e do
próprio solver, compara com z estático (objetivo real) e gera os gráficos.

Calibração (tudo a partir de dados, nenhum valor escolhido à mão):
  1. Singletons de saída (z): quantis de z dos 25% melhores trials de
     results/optuna/trials.csv (q10, q50, q75, q90 -> R3, R1, R4, R2).
  2. Termos de Diversidade/Estagnação: percentis das variáveis de estado
     observadas em execuções estáticas (z = baseline) com sementes de calibração.
  3. Janela de estagnação: percentil 80 das gerações sem melhora observadas.
A avaliação usa sementes diferentes das de calibração (sem vazamento).
"""
from __future__ import annotations
import argparse, csv, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ctypes as ct
from smalib import (FuzzyParams, OUT, build_lib, instance, make_fuzzy, objectives, run_solver)

FUZZY = OUT / "fuzzy"; FUZZY.mkdir(parents=True, exist_ok=True)

def stagnation_counts(hist):
    c, out = 0, []
    for t in range(len(hist)):
        c = 0 if (t == 0 or hist[t] < hist[t - 1]) else c + 1
        out.append(c)
    return np.array(out)

def calibrate(lib, real, insts, pop, iters, z_base, n_seeds):
    trials = pd.read_csv(OUT / "optuna/trials.csv")
    trials = trials[trials.state == "COMPLETE"].sort_values("value")
    k = int(np.ceil(0.25 * len(trials)))
    top = trials.head(k)
    q = np.quantile(top.params_z, [0.10, 0.50, 0.75, 0.90])
    out = [q[1], q[3], q[0], q[2]]            # R1, R2, R3, R4
    div, raw = [], []
    for p in insts.values():
        for s in range(n_seeds):
            r = run_solver(lib, p, real, pop, iters, 100 + s, z_base)
            div += list(r["diversity"]); raw += list(stagnation_counts(r["hist"]))
    div = np.array(div); raw = np.array(raw, float)
    window = float(np.clip(round(np.percentile(raw, 80) / iters, 2), 0.10, 1.0))
    stag = np.clip(raw / max(10.0, window * iters), 0, 1)
    pc = lambda a, lo, hi: [round(float(np.percentile(a, lo)), 3), round(float(np.percentile(a, hi)), 3)]
    cal = {"fonte_saida": "results/optuna/trials.csv", "trials_total": int(len(trials)),
           "trials_top25": int(k), "z_top25_quantis_q10_q50_q75_q90": [float(x) for x in q],
           "out": [round(float(x), 4) for x in out],
           "div_low": pc(div, 10, 60), "div_high": pc(div, 40, 90),
           "stag_low": pc(stag, 10, 60), "stag_high": pc(stag, 40, 90),
           "stag_window": window,
           "diversidade_percentis_P10_P50_P90": [round(float(x), 3) for x in np.percentile(div, [10, 50, 90])],
           "estagnacao_bruta_percentis_P25_P50_P80": [float(x) for x in np.percentile(raw, [25, 50, 80])],
           "sementes_calibracao": f"100..{99 + n_seeds}", "populacao": pop, "iteracoes": iters, "z_execucao_estatica": z_base}
    return cal

def plot_memberships(cal):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    def ramp(v, r, up): return np.clip((v - r[0]) / (r[1] - r[0]), 0, 1) if up else np.clip((r[1] - v) / (r[1] - r[0]), 0, 1)
    for a, key, name in ((ax[0], "div", "Diversidade (D)"), (ax[1], "stag", "Estagnação (E)")):
        hi = max(cal[key + "_high"][1], cal[key + "_low"][1]) * 1.25
        v = np.linspace(0, min(1.0, hi) if key == "div" else 1.0, 400)
        a.plot(v, ramp(v, cal[key + "_low"], False), label="Baixa"); a.plot(v, ramp(v, cal[key + "_high"], True), label="Alta")
        a.set_xlabel(name); a.set_ylabel("Pertinência μ"); a.set_title(f"Termos de {name.split()[0]}"); a.grid(alpha=.3); a.legend()
    names = ["R1\nD baixa,E baixa", "R2\nD baixa,E alta", "R3\nD alta,E baixa", "R4\nD alta,E alta"]
    bars = ax[2].bar(names, cal["out"]); ax[2].bar_label(bars, fmt="%.4f")
    ax[2].set_ylabel("z (singleton)"); ax[2].set_title("Saída Sugeno por regra"); ax[2].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(FUZZY / "funcoes_pertinencia.png", dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--calib-seeds", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=30)
    a = ap.parse_args()
    lib = build_lib(); real = objectives(lib)["real"]
    insts = {"ta001": instance(20, 5, 873654221), "ta011": instance(20, 10, 587595453)}
    optuna = json.loads((OUT / "optuna/resultado_optuna.json").read_text())
    z_base, z_opt = optuna["baseline"].get("z", 0.03), optuna["best_params"]["z"]
    cal = calibrate(lib, real, insts, a.population, a.iterations, 0.03, a.calib_seeds)
    (FUZZY / "calibracao_optuna.json").write_text(json.dumps(cal, indent=2, ensure_ascii=False))
    fp = make_fuzzy(cal["div_low"], cal["div_high"], cal["stag_low"], cal["stag_high"], cal["out"], cal["stag_window"])
    dflt = FuzzyParams(); lib.sma_fuzzy_defaults(ct.byref(dflt))
    same = all(abs(getattr(dflt, k)[i] - getattr(fp, k)[i]) < 1e-9 for k in ("div_low", "div_high", "stag_low", "stag_high", "out") for i in range(len(getattr(fp, k)))) and abs(dflt.stag_window - fp.stag_window) < 1e-9
    cal["defaults_em_C_coincidem_com_calibracao"] = bool(same)
    if not same: print("AVISO: sma_fuzzy_defaults (src/sma.c) difere da calibração; atualize src/sma.c")
    (FUZZY / "calibracao_optuna.json").write_text(json.dumps(cal, indent=2, ensure_ascii=False))
    plot_memberships(cal)

    cfgs = {"Estático baseline (z=0,03)": dict(z=0.03, pop=a.population, fz=None),
            f"Estático Optuna (z={z_opt:.4f})": dict(z=z_opt, pop=a.population, fz=None),
            "Fuzzy adaptativo": dict(z=0.03, pop=a.population, fz=fp),
            f"Ref.: Optuna completo (pop={optuna['best_params']['population']}, z={z_opt:.4f})":
                dict(z=z_opt, pop=optuna["best_params"]["population"], fz=None)}
    rows, finals, curves = [], {}, {}
    for iname, p in insts.items():
        for cname, c in cfgs.items():
            vals, hs, zs = [], [], []
            for s in range(a.eval_seeds):
                r = run_solver(lib, p, real, c["pop"], a.iterations, 5000 + s, c["z"], c["fz"])
                vals.append(r["best"]); hs.append(r["hist"]); zs.append(r["z"].mean())
            finals[(iname, cname)] = np.array(vals); curves[(iname, cname)] = np.mean(hs, axis=0)
            rows.append({"instancia": iname, "configuracao": cname, "populacao": c["pop"],
                         "avaliacoes": c["pop"] * a.iterations, "execucoes": a.eval_seeds,
                         "makespan_medio": float(np.mean(vals)), "desvio": float(np.std(vals, ddof=1)),
                         "melhor": float(np.min(vals)), "pior": float(np.max(vals)), "z_medio": float(np.mean(zs))})
    keys = list(cfgs)
    tests = []
    for iname in insts:
        for ref in (keys[0], keys[1]):
            d = finals[(iname, keys[2])] - finals[(iname, ref)]
            try: pv = float(wilcoxon(d).pvalue)
            except ValueError: pv = float("nan")
            tests.append({"instancia": iname, "fuzzy_vs": ref, "diferenca_media_makespan": float(d.mean()),
                          "wilcoxon_p": pv, "vitorias_fuzzy": int((d < 0).sum()), "empates": int((d == 0).sum()), "derrotas": int((d > 0).sum())})
    with (FUZZY / "comparacao_fuzzy_estatico.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    pd.DataFrame(tests).to_csv(FUZZY / "testes_pareados_fuzzy.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a_, iname in zip(ax, insts):
        for cname in keys[:3]: a_.plot(curves[(iname, cname)], label=cname)
        a_.set_title(f"{iname}: melhor makespan médio ({a.eval_seeds} execuções)"); a_.set_xlabel("Geração"); a_.set_ylabel("Makespan"); a_.grid(alpha=.3); a_.legend()
        lo = min(curves[(iname, k)][-1] for k in keys[:3]); a_.set_ylim(lo - 40, curves[(iname, keys[0])][20])
    plt.tight_layout(); plt.savefig(FUZZY / "comparacao_convergencia_fuzzy.png", dpi=160); plt.close()

    # Execução ilustrativa (objetivo real): trajetória de z, D e E por geração.
    hist_rows = []
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for col, (iname, p) in enumerate(insts.items()):
        r = run_solver(lib, p, real, a.population, a.iterations, 5000, 0.03, fp)
        for g in range(a.iterations):
            hist_rows.append({"instancia": iname, "geracao": g + 1, "melhor_fitness": r["hist"][g], "z": r["z"][g],
                              "diversidade": r["diversity"][g], "estagnacao": r["stagnation"][g]})
        g = np.arange(1, a.iterations + 1)
        axs[0, col].plot(g, r["z"], color="C3", label="z (fuzzy)"); axs[0, col].axhline(0.03, ls="--", c="gray", label="z estático = 0,03")
        axs[0, col].set_ylabel("z (taxa de reinicialização)"); axs[0, col].set_title(f"{iname} — parâmetro ajustado pelo controlador fuzzy"); axs[0, col].legend(); axs[0, col].grid(alpha=.3)
        axs[1, col].plot(g, r["diversity"]); axs[1, col].set_ylabel("Diversidade D"); axs[1, col].grid(alpha=.3)
        axs[2, col].plot(g, r["stagnation"]); axs[2, col].set_ylabel("Estagnação E"); axs[2, col].set_xlabel("Geração"); axs[2, col].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FUZZY / "variacao_z_geracoes.png", dpi=160); plt.close()
    pd.DataFrame(hist_rows).to_csv(FUZZY / "historico_controlador.csv", index=False)
    zs = pd.DataFrame(hist_rows)
    stats = {k: {"z_min": float(g.z.min()), "z_max": float(g.z.max()), "z_desvio": float(g.z.std()),
                 "z_distintos_6casas": int(g.z.round(6).nunique()),
                 "fracao_z_em_singleton_R1": float((abs(g.z - cal["out"][0]) < 1e-9).mean())} for k, g in zs.groupby("instancia")}
    (FUZZY / "resumo_fuzzy.json").write_text(json.dumps({"calibracao": cal, "comparacao": rows, "testes": tests, "variacao_z": stats}, indent=2, ensure_ascii=False))
    print(json.dumps({"calibracao": cal, "variacao_z": stats}, indent=2, ensure_ascii=False))
    print(pd.DataFrame(rows)[["instancia", "configuracao", "makespan_medio", "desvio", "melhor"]].to_string())
    print(pd.DataFrame(tests).to_string())

if __name__ == "__main__": main()
