#!/usr/bin/env python3
"""Utilidades compartilhadas (ctypes + PFS) pelos scripts das Semanas 4 e 5."""
from __future__ import annotations
import ctypes as ct, subprocess, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "src", ROOT / "results"

class Pfs(ct.Structure):
    _fields_ = [("jobs", ct.c_int), ("machines", ct.c_int), ("p", (ct.c_int * 500) * 20)]
class Config(ct.Structure):
    _fields_ = [("population", ct.c_int), ("iterations", ct.c_int), ("dimension", ct.c_int),
                ("lower", ct.c_double), ("upper", ct.c_double), ("z", ct.c_double),
                ("seed", ct.c_uint64)]
class FuzzyParams(ct.Structure):
    _fields_ = [("div_low", ct.c_double * 2), ("div_high", ct.c_double * 2),
                ("stag_low", ct.c_double * 2), ("stag_high", ct.c_double * 2),
                ("out", ct.c_double * 4), ("stag_window", ct.c_double)]
Objective = ct.CFUNCTYPE(ct.c_double, ct.POINTER(ct.c_double), ct.c_int, ct.c_void_p)

def make_fuzzy(div_low, div_high, stag_low, stag_high, out, stag_window) -> FuzzyParams:
    fp = FuzzyParams()
    for name, v in (("div_low", div_low), ("div_high", div_high),
                    ("stag_low", stag_low), ("stag_high", stag_high), ("out", out)):
        for i, x in enumerate(v): getattr(fp, name)[i] = float(x)
    fp.stag_window = float(stag_window)
    return fp

def instance(jobs, machines, seed):
    """Gerador de Taillard (Park-Miller); idêntico a pfs_generate em C."""
    p = np.zeros((machines, jobs), dtype=np.int32)
    for m in range(machines):
        for j in range(jobs):
            seed = (16807 * seed) % 2147483647
            p[m, j] = 1 + int(99.0 * (seed / 2147483647.0))
    return p

def decode(x):
    return np.argsort(x, kind="stable")

def makespan(p, perm):
    m, _ = p.shape
    end = [0] * m
    for job in perm:
        prev = 0
        for k in range(m):
            end[k] = max(end[k], prev) + int(p[k, job])
            prev = end[k]
    return float(end[-1])

def build_lib():
    (ROOT / "build").mkdir(exist_ok=True)
    srcs = ["src/sma.c", "src/pfs.c", "src/rng.c"]
    if (SRC / "surrogate_model.c").exists(): srcs.append("src/surrogate_model.c")
    subprocess.run(["cc", "-O2", "-std=c11", "-fPIC", "-shared", "-Iinclude", *srcs,
                    "-lm", "-o", "build/libsmalab.so"], cwd=ROOT, check=True)
    lib = ct.CDLL(str(ROOT / "build/libsmalab.so"))
    dp = ct.POINTER(ct.c_double)
    lib.sma_solve_ex.argtypes = [ct.POINTER(Config), Objective, ct.c_void_p, dp, dp, dp,
                                 ct.POINTER(FuzzyParams), dp, dp, dp]
    lib.sma_fuzzy_defaults.argtypes = [ct.POINTER(FuzzyParams)]
    lib.pfs_set_cost_repeat.argtypes = [ct.c_int]
    if hasattr(lib, "surrogate_objective"):
        lib.surrogate_objective.argtypes = [dp, ct.c_int, ct.c_void_p]
        lib.surrogate_objective.restype = ct.c_double
    return lib

def objectives(lib):
    return {"real": ct.cast(lib.pfs_objective, Objective),
            "real_caro": ct.cast(lib.pfs_objective_costly, Objective),
            "surrogate": ct.cast(lib.surrogate_objective, Objective)}

def run_solver(lib, p, callback, pop=30, iterations=300, seed=1, z=0.03, fuzzy=None):
    """Executa o SMA. fuzzy=None => z fixo; senão controlador adaptativo.
    Sempre registra z, diversidade e estagnação por geração."""
    n = 20
    pc = Pfs(); pc.jobs = n; pc.machines = p.shape[0]
    for m in range(p.shape[0]):
        for j in range(n): pc.p[m][j] = int(p[m, j])
    cfg = Config(pop, iterations, n, -1.0, 1.0, z, seed)
    bestx = (ct.c_double * n)(); hist = (ct.c_double * iterations)(); best = ct.c_double()
    zh, dh, sh = ((ct.c_double * iterations)() for _ in range(3))
    ctx = ct.cast(ct.pointer(pc), ct.c_void_p)
    fp = ct.byref(fuzzy) if fuzzy is not None else None
    start = time.perf_counter()
    rc = lib.sma_solve_ex(ct.byref(cfg), callback, ctx, bestx, hist, ct.byref(best), fp, zh, dh, sh)
    elapsed = time.perf_counter() - start
    if rc: raise RuntimeError("sma_solve_ex retornou erro")
    return {"best": best.value, "time": elapsed, "hist": np.array(hist), "z": np.array(zh),
            "diversity": np.array(dh), "stagnation": np.array(sh), "x": np.array(list(bestx))}
