"""Full experiment suite for the CAiSE 2027 paper.

E1  cross-edition identification (t -> t+1), baselines + identified variants
E2  semi-synthetic non-monotonic criterion (two scenarios)
E3  fuzzy (min, median, max) estimates: width and coverage vs n
E4  penalty strength (lambda) sensitivity
E5  rank reversal: FN-MABAC vs classic MABAC / TOPSIS when alternatives change

Usage: python3 -I full_experiments.py <E1|E2|E3|E4|E5> <csv_dir> <out_dir> <workers>
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WHICH, CSV, OUT, WORKERS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.argv = ["experiment.py", CSV, "2025", OUT, "1"]

import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np
import pandas as pd
from mealpy import FloatVar, PSO, DE
import pymcdm.weights as PW
from pymcdm.methods import MABAC, TOPSIS
import experiment as e

M, LO, HI = e.M, e.LO, e.HI
REPS = 30
NS = [5, 10, 20, 40]
NONE, ALL = [False] * 5, [True] * 5
PAIRS = [(2023, 2024), (2024, 2025), (2025, 2026), (2026, 2027)]


# ------------------------------------------------------------ knowledge
def kn_correct(w):
    T, RE, RQ, IND, IO = w
    v = sum(max(0.0, IND - x) for x in (T, RE, RQ, IO))
    v += sum(max(0.0, IO - x) for x in (T, RE, RQ))
    v += sum(max(0.0, 0.2 - x) for x in (T, RE, RQ))
    return v


def kn_wrong(w):
    T, RE, RQ, IND, IO = w
    return max(0.0, 0.2 - IND) + max(0.0, RQ - 0.1)


def kn_none(w):
    return 0.0


KN = {"none": kn_none, "correct": kn_correct, "wrong": kn_wrong}


def identify(X, ref, learn, kn, seed, lam=1.0, core_floor=None, w_syn=None, opt="pso"):
    learn = np.asarray(learn, bool)
    k = int(learn.sum())
    lb = [1e-3] * M + [LO + 1.0] * k
    ub = [1.0] * M + [HI] * k
    viol = KN[kn]

    def decode(sol):
        w = np.asarray(sol[:M], float); w = w / w.sum()
        c = np.full(M, HI); c[learn] = np.asarray(sol[M:], float)
        return w, c

    def fit(sol):
        w, c = decode(sol)
        v = viol(w)
        if core_floor is not None and k:
            v += np.sum(np.maximum(0.0, core_floor - c[learn])) / 100.0
        return e.rw(e.fn_mabac(X, w, c), ref) - lam * v

    model = PSO.OriginalPSO(epoch=e.EPOCH, pop_size=e.POP)
    if opt == "jade":
        model = DE.JADE(epoch=300, pop_size=50, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1)
    best = model.solve({"obj_func": fit, "bounds": FloatVar(lb=lb, ub=ub),
                        "minmax": "max", "log_to": None}, seed=seed)
    return decode(best.solution)


_CACHE = {}


def data(year):
    if year not in _CACHE:
        d = e.load(year)
        _CACHE[year] = (d, d[e.PILLARS].to_numpy(float), d["Overall"].to_numpy())
    return _CACHE[year]


def row_eval(**kw):
    return kw


# ------------------------------------------------------------ E1
E1_IDENT = {
    "SITW": (NONE, "none", None),
    "SITW + knowledge": (NONE, "correct", None),
    "SITW + wrong knowledge": (NONE, "wrong", None),
    "All cores free": (ALL, "none", None),
    "All cores + knowledge": (ALL, "correct", 90.0),
}


def e1_task(args):
    (y_tr, y_te), n, r = args
    dtr, Xtr_all, otr = data(y_tr)
    dte, Xte, ote = data(y_te)
    ref_te = e.rank_desc(ote)
    new = ~dte.Name.isin(set(dtr.Name)).to_numpy()
    ref_new = e.rank_desc(ote[new])
    idx = np.random.default_rng(10_000 * y_tr + 100 * n + r).permutation(len(dtr))[:n]
    Xs, rs = Xtr_all[idx], e.rank_desc(otr[idx])
    types = np.ones(M)
    cand = {
        "Equal": (PW.equal_weights(Xtr_all), np.full(M, HI)),
        "Entropy": (PW.entropy_weights(Xtr_all), np.full(M, HI)),
        "CRITIC": (PW.critic_weights(Xtr_all), np.full(M, HI)),
        "Gini": (PW.gini_weights(Xtr_all), np.full(M, HI)),
    }
    for name, (mask, kn, floor) in E1_IDENT.items():
        cand[name] = identify(Xs, rs, mask, kn, seed=r, core_floor=floor)
    out = []
    for name, (w, c) in cand.items():
        out.append(dict(pair=f"{y_tr}->{y_te}", n_train=n, rep=r, method=name,
                        rw_train=e.rw(e.fn_mabac(Xs, w, c), rs),
                        rw_next=e.rw(e.fn_mabac(Xte, w, c), ref_te),
                        rw_next_new=e.rw(e.fn_mabac(Xte[new], w, c), ref_new),
                        n_new=int(new.sum()),
                        w_mae=np.abs(w - e.W_TRUE).mean(),
                        core_mae=np.abs(c - e.CORE_TRUE).mean(),
                        **{f"w_{s}": w[i] for i, s in enumerate(e.SHORT)}))
    return out


# ------------------------------------------------------------ E2
SCEN = {
    "IO core 75": (np.array([0.25, 0.25, 0.25, 0.04, 0.21]),
                   np.array([100, 100, 100, 100, 75.0]), 4),
    "T core 70": (np.array([0.30, 0.25, 0.25, 0.05, 0.15]),
                  np.array([70.0, 100, 100, 100, 100]), 0),
}


def e2_task(args):
    scen, n, r = args
    w_syn, c_syn, j = SCEN[scen]
    d, X, _ = data(2025)
    d2, X2, _ = data(2026)                  # apply to next edition as test
    pref_tr, pref_te = e.fn_mabac(X, w_syn, c_syn), e.fn_mabac(X2, w_syn, c_syn)
    idx = np.random.default_rng(50_000 + 100 * n + r).permutation(len(d))[:n]
    ref_tr = e.rank_desc(pref_tr[idx]); ref_te = e.rank_desc(pref_te)
    correct = [False] * 5; correct[j] = True
    wrong = [False] * 5; wrong[3 if j != 3 else 1] = True
    configs = {
        "No flag": (NONE, None),
        "Correct flag": (correct, None),
        "Wrong flag": (wrong, None),
        "All cores free": (ALL, None),
        "All cores, monotonic belief": (ALL, 90.0),
    }
    out = []
    for name, (mask, floor) in configs.items():
        w, c = identify(X[idx], ref_tr, mask, "none", seed=r, core_floor=floor)
        out.append(dict(scenario=scen, n_train=n, rep=r, method=name,
                        rw_train=e.rw(e.fn_mabac(X[idx], w, c), ref_tr),
                        rw_test=e.rw(e.fn_mabac(X2, w, c), ref_te),
                        w_mae=np.abs(w - w_syn).mean(),
                        core_err=abs(c[j] - c_syn[j]),
                        core_learned=c[j]))
    return out


# ------------------------------------------------------------ E3
def e3_task(args):
    n, tset, kn = args
    d, X, o = data(2026)
    idx = np.random.default_rng(70_000 + 100 * n + tset).permutation(len(d))[:n]
    ref = e.rank_desc(o[idx])
    W = np.array([identify(X[idx], ref, NONE, kn, seed=1000 + s)[0] for s in range(15)])
    lo, med, hi = W.min(0), np.median(W, 0), W.max(0)
    out = []
    for i, s in enumerate(e.SHORT):
        out.append(dict(n_train=n, tset=tset, knowledge=kn, criterion=s,
                        lo=lo[i], med=med[i], hi=hi[i], width=hi[i] - lo[i],
                        true=e.W_TRUE[i],
                        covered=bool(lo[i] - 1e-9 <= e.W_TRUE[i] <= hi[i] + 1e-9),
                        abs_err_median=abs(med[i] - e.W_TRUE[i])))
    return out


# ------------------------------------------------------------ E4
def e4_task(args):
    lam, n, r, kn = args
    dtr, Xtr, otr = data(2026)
    dte, Xte, ote = data(2027)
    idx = np.random.default_rng(10_000 * 2026 + 100 * n + r).permutation(len(dtr))[:n]
    w, c = identify(Xtr[idx], e.rank_desc(otr[idx]), NONE, kn, seed=r, lam=lam)
    return [dict(lam=lam, n_train=n, rep=r, knowledge=kn,
                 rw_train=e.rw(e.fn_mabac(Xtr[idx], w, c), e.rank_desc(otr[idx])),
                 rw_next=e.rw(e.fn_mabac(Xte, w, c), e.rank_desc(ote)),
                 w_mae=np.abs(w - e.W_TRUE).mean())]


# ------------------------------------------------------------ E5
def e5():
    d, X, o = data(2027)
    types = np.ones(M)
    meths = {
        "FN-MABAC": lambda Z: e.fn_mabac(Z, e.W_TRUE, e.CORE_TRUE),
        "MABAC": lambda Z: MABAC()(Z, e.W_TRUE, types),
        "TOPSIS": lambda Z: TOPSIS()(Z, e.W_TRUE, types),
    }
    rows = []
    for r in range(200):
        rng = np.random.default_rng(90_000 + r)
        perm = rng.permutation(len(d))
        base, extra = perm[:50], perm[50:100]
        for mname, f in meths.items():
            p0 = f(X[base])
            for scenario, idx in [("add 50", np.concatenate([base, extra])),
                                  ("remove 10", base[10:])]:
                p1 = f(X[idx])
                keep = base if scenario == "add 50" else base[10:]
                a = p0[:len(keep)] if scenario == "add 50" else p0[10:]
                b = p1[:len(keep)]
                diff_a = np.sign(a[:, None] - a[None, :])
                diff_b = np.sign(b[:, None] - b[None, :])
                iu = np.triu_indices(len(keep), 1)
                flips = np.mean(diff_a[iu] != diff_b[iu])
                rows.append(dict(rep=r, method=mname, scenario=scenario,
                                 pair_flip_rate=flips,
                                 any_reversal=bool(flips > 0)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/E5_rank_reversal.csv", index=False)
    print(df.groupby(["scenario", "method"])[["pair_flip_rate", "any_reversal"]]
            .mean().round(5).to_string(), flush=True)


# ------------------------------------------------------------ driver
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
    if WHICH == "E1":
        run("E1_cross_edition", e1_task,
            [(p, n, r) for p in PAIRS for n in NS for r in range(REPS)])
    elif WHICH == "E2":
        run("E2_nonmonotonic", e2_task,
            [(s, n, r) for s in SCEN for n in [10, 20, 40] for r in range(REPS)])
    elif WHICH == "E3":
        run("E3_fuzzy", e3_task,
            [(n, t, kn) for n in NS for t in range(5) for kn in ["none", "correct"]])
    elif WHICH == "E4":
        run("E4_lambda", e4_task,
            [(lam, n, r, kn) for lam in [0.1, 0.5, 1.0, 5.0] for n in [5, 10]
             for r in range(REPS) for kn in ["correct", "wrong"]])
    elif WHICH == "E5":
        e5()
        print("E5 DONE", flush=True)
