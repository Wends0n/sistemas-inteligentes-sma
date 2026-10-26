#!/usr/bin/env python3
"""Otimiza hiperparâmetros estáticos do SMA-PFS com Optuna.

O objetivo é a média do makespan final em ta001 e ta011, com sementes fixas.
Cada trial usa o mesmo orçamento (execuções x iterações x população variável).
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "build" / "sma_pfs"
OUT = ROOT / "results" / "optuna"

def evaluate(population: int, z: float, runs: int, iterations: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="sma_optuna_") as tmp:
        outdir = Path(tmp)
        subprocess.run(
            [str(SOLVER), str(outdir), str(runs), str(iterations),
             str(population), f"{z:.8f}"],
            cwd=ROOT, check=True, capture_output=True, text=True
        )
        with (outdir / "runs.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    by_instance = {}
    for name in ("ta001", "ta011"):
        vals = [float(r["best_makespan"]) for r in rows if r["instance"] == name]
        if len(vals) != runs:
            raise RuntimeError(f"Esperadas {runs} execuções para {name}; obtidas {len(vals)}")
        by_instance[name] = sum(vals) / len(vals)
    return {
        "objective": sum(by_instance.values()) / len(by_instance),
        "ta001_mean": by_instance["ta001"],
        "ta011_mean": by_instance["ta011"],
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3,
                        help="execuções por instância em cada avaliação (padrão: 3)")
    parser.add_argument("--iterations", type=int, default=200,
                        help="iterações por execução (padrão: 200)")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.trials < 2 or args.runs < 1 or args.iterations < 1:
        parser.error("trials deve ser >= 2; runs e iterations devem ser >= 1")

    subprocess.run(["make", "all"], cwd=ROOT, check=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # Baseline avaliado com o mesmo orçamento de cada trial.
    baseline = evaluate(population=30, z=0.03, runs=args.runs,
                        iterations=args.iterations)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name="sma_pfs_static_hyperparameters")
    def objective(trial):
        population = trial.suggest_int("population", 10, 60, step=5)
        z = trial.suggest_float("z", 0.0, 0.10)
        result = evaluate(population, z, args.runs, args.iterations)
        trial.set_user_attr("ta001_mean", result["ta001_mean"])
        trial.set_user_attr("ta011_mean", result["ta011_mean"])
        return result["objective"]

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    best = study.best_trial
    tuned = evaluate(int(best.params["population"]), float(best.params["z"]),
                     args.runs, args.iterations)

    # Gráficos exigidos: histórico e importância de hiperparâmetros.
    trials = [t for t in study.trials if t.value is not None]
    plt.figure(figsize=(9, 5))
    plt.plot([t.number + 1 for t in trials], [t.value for t in trials],
             marker="o", linewidth=1, label="Valor de cada trial")
    running = []
    best_so_far = float("inf")
    for t in trials:
        best_so_far = min(best_so_far, float(t.value))
        running.append(best_so_far)
    plt.plot([t.number + 1 for t in trials], running,
             linewidth=2, label="Melhor valor acumulado")
    plt.xlabel("Trial")
    plt.ylabel("Makespan médio final (menor é melhor)")
    plt.title("Optuna — histórico de otimização SMA-PFS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "historico_otimizacao.png", dpi=160)
    plt.close()

    try:
        import optuna.visualization.matplotlib as ovm
        ax = ovm.plot_param_importances(study)
        ax.set_title("Importância dos hiperparâmetros — Optuna")
        ax.figure.tight_layout()
        ax.figure.savefig(OUT / "importancia_hiperparametros.png", dpi=160)
        plt.close(ax.figure)
    except Exception as exc:
        # Fallback gráfico sem depender do backend experimental do Optuna.
        from optuna.importance import get_param_importances
        importance = get_param_importances(study)
        names, values = list(importance.keys()), list(importance.values())
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(names, values)
        ax.set_xlabel("Importância relativa")
        ax.set_title("Importância dos hiperparâmetros — Optuna")
        fig.tight_layout()
        fig.savefig(OUT / "importancia_hiperparametros.png", dpi=160)
        plt.close(fig)
        print(f"Aviso: gráfico via fallback ({exc})")

    comparison = [
        {"configuracao": "Baseline", "population": 30, "z": 0.03,
         "ta001_makespan_medio": baseline["ta001_mean"],
         "ta011_makespan_medio": baseline["ta011_mean"],
         "objetivo_medio": baseline["objective"]},
        {"configuracao": "Tunada (Optuna)", "population": int(best.params["population"]),
         "z": float(best.params["z"]), "ta001_makespan_medio": tuned["ta001_mean"],
         "ta011_makespan_medio": tuned["ta011_mean"], "objetivo_medio": tuned["objective"]},
    ]
    with (OUT / "comparacao_baseline_tunada.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)
    with (OUT / "resultado_optuna.json").open("w", encoding="utf-8") as f:
        json.dump({
            "study": study.study_name, "trials": args.trials,
            "runs_per_instance": args.runs, "iterations": args.iterations,
            "seed": args.seed, "best_trial": best.number,
            "best_params": best.params, "best_value": best.value,
            "baseline": baseline, "tuned": tuned,
            "comparison": comparison,
        }, f, indent=2, ensure_ascii=False)
    study.trials_dataframe().to_csv(OUT / "trials.csv", index=False)
    print("\nMelhores parâmetros:", best.params)
    print("Baseline:", baseline)
    print("Tunada:", tuned)
    print("Arquivos salvos em:", OUT)

if __name__ == "__main__":
    main()
