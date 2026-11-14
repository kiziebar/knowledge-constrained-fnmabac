"""E8: real non-monotonic case -- graphics cards (EAAI 2025 data, Tom's Hardware ranking).

Usage: python3 -I gpu_experiment.py <CHECK|SPLIT|FULL> <out_dir> <workers>
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WHICH, OUT, WORKERS = sys.argv[1], sys.argv[2], int(sys.argv[3])

import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import linprog
from mealpy import FloatVar, PSO, DE
from pymcdm.correlations import weighted_spearman
import gpu_data as G

M = 8
EPOCH, POP = 200, 50
REPS = 30


def membership(X, cores):
    N = np.empty_like(X)
    for j in range(M):
        a, b, c = G.LO[j], G.HI[j], cores[j]
        x = X[:, j]
        left = np.where(c > a, (x - a) / max(c - a, 1e-12), (x <= a).astype(float))
        right = np.where(c < b, (b - x) / max(b - c, 1e-12), (x >= b).astype(float))
        N[:, j] = np.where(x <= c, left, right)
    return np.clip(N, 0, 1)


def fn_mabac(X, w, cores):
    V = w * (membership(X, cores) + 1)
    G_ = np.prod(V, axis=0) ** (1 / V.shape[0])
    return (V - G_).sum(axis=1)


def rank_desc(p):
    return rankdata(-np.asarray(p), method="average")


def rw(pref, ref_rank):
    return weighted_spearman(rank_desc(pref), ref_rank)


# ------------------------------------------------------------ knowledge
# Expert view: the reference is a *performance* hierarchy, so price should carry
# little weight, and raw compute resources (shader units C6, TMUs C7) matter more
# than clock frequencies (C4, C5).
def kn_correct(w):
    v = max(0.0, w[0] - 0.10)
    v += sum(max(0.0, w[k] - w[j]) for j in (5, 6) for k in (3, 4))
    return v


def kn_wrong(w):           # 'price dominates, shader count is irrelevant'
    return max(0.0, 0.30 - w[0]) + max(0.0, w[5] - 0.05)


def kn_none(w):
    return 0.0


KN = {"none": kn_none, "correct": kn_correct, "wrong": kn_wrong}


def identify(X, ref, learn, kn, seed, fixed_w=None, lam=1.0, opt="pso"):
    learn = np.asarray(learn, bool)
    k = int(learn.sum())
    nw = 0 if fixed_w is not None else M
    lb = [1e-3] * nw + list(G.LO[learn])
    ub = [1.0] * nw + list(G.HI[learn])
    viol = KN[kn]

    def decode(sol):
        sol = np.asarray(sol, float)
        if fixed_w is None:
            w = sol[:M] / sol[:M].sum()
        else:
            w = fixed_w
        c = G.CORE_TYPE.copy()
        c[learn] = sol[nw:]
        return w, c

    def fit(sol):
        w, c = decode(sol)
        return rw(fn_mabac(X, w, c), ref) - lam * viol(w)

    model = PSO.OriginalPSO(epoch=EPOCH, pop_size=POP)
    if opt == "jade":
        model = DE.JADE(epoch=300, pop_size=50, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1)
    best = model.solve({"obj_func": fit, "bounds": FloatVar(lb=lb, ub=ub),
                        "minmax": "max", "log_to": None}, seed=seed)
    return decode(best.solution)


def lp_weights(X, ref, knowledge=True, C=10.0):
    N = membership(X, G.CORE_TYPE)
    order = np.argsort(ref)
    pairs = [(order[i], order[k]) for i in range(len(order)) for k in range(i + 1, len(order))
             if ref[order[i]] < ref[order[k]]]
    P = len(pairs); nv = M + 1 + P
    c = np.zeros(nv); c[M] = -1.0; c[M + 1:] = C
    A, b = [], []
    for p, (a, z) in enumerate(pairs):
        row = np.zeros(nv); row[:M] = -(N[a] - N[z]); row[M] = 1.0; row[M + 1 + p] = -1.0
        A.append(row); b.append(0.0)
    if knowledge:
        row = np.zeros(nv); row[0] = 1; A.append(row); b.append(0.10)
        for j in (5, 6):
            for k in (3, 4):
                row = np.zeros(nv); row[k] = 1; row[j] = -1; A.append(row); b.append(0.0)
    Aeq = np.zeros((1, nv)); Aeq[0, :M] = 1
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[1.0],
                  bounds=[(0, 1)] * M + [(None, 1.0)] + [(0, None)] * P, method="highs")
    w = np.clip(res.x[:M], 0, None) if res.success else np.full(M, 1 / M)
    return w / w.sum()


FLAG_C3 = [False, False, True] + [False] * 5
FLAG_C4 = [False] * 3 + [True] + [False] * 4
NONE = [False] * M
ALL = [True] * M

CONFIGS = {
    # name: (learn mask, knowledge, fixed weights)
    "SITW": (NONE, "none", None),
    "SITW + knowledge": (NONE, "correct", None),
    "SITW + wrong knowledge": (NONE, "wrong", None),
    "Flag C3": (FLAG_C3, "none", None),
    "Flag C3 + knowledge": (FLAG_C3, "correct", None),
    "Wrong flag C4 + knowledge": (FLAG_C4, "correct", None),
    "All cores + weights": (ALL, "none", None),
    "STFN-MABAC (Gini w, all cores)": (ALL, "none", "gini"),
}


def evaluate(name, w, c, tr, te, ref_tr, ref_te, n, r):
    return dict(n_train=n, rep=r, method=name,
                rw_train=rw(fn_mabac(G.X[tr], w, c), ref_tr),
                rw_test=rw(fn_mabac(G.X[te], w, c), ref_te),
                core_C3=c[2], **{f"w_{s}": w[k] for k, s in enumerate(G.SHORT)})


def split_task(args):
    n, r = args
    idx = np.random.default_rng(60_000 + 100 * n + r).permutation(30)
    tr, te = idx[:n], idx[n:]
    ref_tr = rankdata(G.REF[tr]); ref_te = rankdata(G.REF[te])
    out = [evaluate("Gini (FN-MABAC)", G.W_GINI, G.CORE_TYPE, tr, te, ref_tr, ref_te, n, r)]
    w = lp_weights(G.X[tr], ref_tr, True)
    out.append(evaluate("LP + knowledge", w, G.CORE_TYPE, tr, te, ref_tr, ref_te, n, r))
    for name, (mask, kn, fw) in CONFIGS.items():
        w, c = identify(G.X[tr], ref_tr, mask, kn, seed=r,
                        fixed_w=G.W_GINI if fw == "gini" else None)
        out.append(evaluate(name, w, c, tr, te, ref_tr, ref_te, n, r))
    return out


def full_task(args):
    name, s = args
    mask, kn, fw = CONFIGS[name]
    ref = G.REF
    w, c = identify(G.X, ref, mask, kn, seed=3000 + s, fixed_w=G.W_GINI if fw == "gini" else None)
    return [dict(method=name, seed=s, rw_train=rw(fn_mabac(G.X, w, c), ref),
                 **{f"w_{x}": w[k] for k, x in enumerate(G.SHORT)},
                 **{f"c_{x}": c[k] for k, x in enumerate(G.SHORT)})]


def run(name, fn, tasks):
    rows = []
    with Pool(WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            rows.extend(res)
            if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(f"{OUT}/{name}.csv", index=False)
                print(f"{name}: {i}/{len(tasks)}", flush=True)
    print(f"{name} DONE", flush=True)


if __name__ == "__main__":
    if WHICH == "CHECK":
        p = fn_mabac(G.X, G.W_GINI, G.CORE_TYPE)
        rk = rank_desc(p)
        print("ranking equals published FN-MABAC:", np.array_equal(rk, G.FN_PUBLISHED))
        print("rw(FN-MABAC Gini, REF) =", round(weighted_spearman(rk, G.REF), 5), "(published 0.92074)")
        c = G.CORE_TYPE.copy(); c[2] = 191.69
        print("rw with C3 core 191.69 and Gini weights =", round(rw(fn_mabac(G.X, G.W_GINI, c), G.REF), 5))
    elif WHICH == "SPLIT":
        run("E8_gpu_split", split_task, [(n, r) for n in [10, 15, 20] for r in range(REPS)])
    elif WHICH == "FULL":
        run("E8_gpu_full", full_task, [(nm, s) for nm in CONFIGS for s in range(15)])
