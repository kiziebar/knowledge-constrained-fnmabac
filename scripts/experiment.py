"""Pilot experiment: knowledge-constrained joint identification of weights
and TFN cores for FN-MABAC on THE World University Rankings data.

Usage: python3 -I experiment.py <csv_dir> <year> <out_dir> [repeats]
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pymcdm.correlations import weighted_spearman
from mealpy import FloatVar, PSO

CSV_DIR, YEAR, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 10

PILLARS = ["Teaching", "Research Environment", "Research Quality",
           "Industry", "International Outlook"]
SHORT = ["T", "RE", "RQ", "IND", "IO"]
W_TRUE = np.array([0.295, 0.29, 0.30, 0.04, 0.075])
CORE_TRUE = np.full(5, 100.0)
LO, HI = 0.0, 100.0
M = len(PILLARS)
EPOCH, POP = 150, 40


# ---------------------------------------------------------------- data
def load(year):
    df = pd.read_csv(f"{CSV_DIR}/THE_{year}_rankings.csv", dtype=str)
    raw = df["Overall"].fillna("")
    df = df[~raw.str.contains("[–-]") & (raw != "")].copy()
    for c in PILLARS + ["Overall"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=PILLARS + ["Overall"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- FN-MABAC
def fuzzy_norm(X, cores):
    """Profit-type TFN (LO, core, HI) membership, per criterion."""
    N = np.empty_like(X)
    for j in range(X.shape[1]):
        c = cores[j]
        x = X[:, j]
        left = (x - LO) / max(c - LO, 1e-9)
        right = (HI - x) / max(HI - c, 1e-9) if c < HI else np.ones_like(x)
        N[:, j] = np.where(x <= c, left, right)
    return np.clip(N, 0, 1)


def fn_mabac(X, w, cores):
    V = w * (fuzzy_norm(X, cores) + 1)
    G = np.prod(V, axis=0) ** (1 / V.shape[0])
    return (V - G).sum(axis=1)


def rank_desc(x):
    return rankdata(-np.asarray(x), method="average")


def rw(pref, ref_rank):
    return weighted_spearman(rank_desc(pref), ref_rank)


# ---------------------------------------------------------------- knowledge
def viol_correct(w, c):
    """Expert knowledge consistent with THE's published narrative
    (ordinal / loose bounds only, no exact weights)."""
    T, RE, RQ, IND, IO = w
    v = 0.0
    v += sum(max(0.0, IND - x) for x in (T, RE, RQ, IO))   # industry smallest
    v += sum(max(0.0, IO - x) for x in (T, RE, RQ))        # IO below big three
    v += sum(max(0.0, 0.2 - x) for x in (T, RE, RQ))       # big three >= 0.2
    v += np.sum(np.maximum(0.0, 90.0 - c)) / 100.0         # monotonic: core >= 90
    return v


def viol_wrong(w, c):
    """Deliberately misspecified expert knowledge."""
    T, RE, RQ, IND, IO = w
    v = 0.0
    v += max(0.0, 0.2 - IND)        # 'industry is a major criterion'
    v += max(0.0, RQ - 0.1)         # 'research quality matters little'
    return v


def viol_none(w, c):
    return 0.0


# ---------------------------------------------------------------- identification
def identify(X, ref_rank, mode, viol, seed, lam=1.0):
    """mode: 'w' (SITW, cores fixed at 100) or 'wc' (joint weights + cores)."""
    if mode == "w":
        bounds = FloatVar(lb=[1e-3] * M, ub=[1.0] * M)
    else:
        bounds = FloatVar(lb=[1e-3] * M + [LO + 1.0] * M,
                          ub=[1.0] * M + [HI] * M)

    def decode(sol):
        w = np.asarray(sol[:M], float)
        w = w / w.sum()
        c = np.asarray(sol[M:], float) if mode == "wc" else CORE_TRUE
        return w, c

    def fitness(sol):
        w, c = decode(sol)
        return rw(fn_mabac(X, w, c), ref_rank) - lam * viol(w, c)

    problem = {"obj_func": fitness, "bounds": bounds, "minmax": "max",
               "log_to": None}
    model = PSO.OriginalPSO(epoch=EPOCH, pop_size=POP)
    best = model.solve(problem, seed=seed)
    return decode(best.solution)


def gini_weights(X):
    G = []
    for j in range(X.shape[1]):
        x = X[:, j]
        G.append(np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) ** 2 * x.mean()))
    G = np.array(G)
    return G / G.sum()


# ---------------------------------------------------------------- main
def main():
    df = load(YEAR)
    X_all = df[PILLARS].to_numpy(float)
    ref_all = rank_desc(df["Overall"].to_numpy())
    n_all = len(df)
    print(f"{YEAR}: {n_all} institutions with exact Overall", flush=True)

    # sanity: true model on all data
    print("true model rw =", round(rw(fn_mabac(X_all, W_TRUE, CORE_TRUE), ref_all), 5))

    methods = {
        "SITW (w only)":          ("w",  viol_none),
        "Joint (w+cores)":        ("wc", viol_none),
        "Joint + knowledge":      ("wc", viol_correct),
        "Joint + wrong knowledge": ("wc", viol_wrong),
    }
    rows = []
    g_w = gini_weights(X_all)
    for n_train in [5, 10, 20, 40]:
        for r in range(REPEATS):
            rng = np.random.default_rng(1000 * n_train + r)
            idx = rng.permutation(n_all)
            tr, te = idx[:n_train], idx[n_train:]
            Xtr, Xte = X_all[tr], X_all[te]
            ref_tr = rank_desc(df["Overall"].to_numpy()[tr])
            ref_te = rank_desc(df["Overall"].to_numpy()[te])
            base = {
                "Equal weights": (np.full(M, 1 / M), CORE_TRUE),
                "Gini weights": (g_w, CORE_TRUE),
            }
            for name, (w, c) in base.items():
                rows.append(dict(n_train=n_train, rep=r, method=name,
                                 rw_test=rw(fn_mabac(Xte, w, c), ref_te),
                                 w_mae=np.abs(w - W_TRUE).mean(),
                                 core_mae=np.abs(c - CORE_TRUE).mean(),
                                 **{f"w_{s}": w[k] for k, s in enumerate(SHORT)}))
            for name, (mode, viol) in methods.items():
                t0 = time.time()
                w, c = identify(Xtr, ref_tr, mode, viol, seed=r)
                rows.append(dict(n_train=n_train, rep=r, method=name,
                                 rw_train=rw(fn_mabac(Xtr, w, c), ref_tr),
                                 rw_test=rw(fn_mabac(Xte, w, c), ref_te),
                                 w_mae=np.abs(w - W_TRUE).mean(),
                                 core_mae=np.abs(c - CORE_TRUE).mean(),
                                 secs=time.time() - t0,
                                 **{f"w_{s}": w[k] for k, s in enumerate(SHORT)},
                                 **{f"c_{s}": c[k] for k, s in enumerate(SHORT)}))
            print(f"n={n_train} rep={r} done", flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/results_{YEAR}.csv", index=False)

    # fuzzy estimates: one fixed training set, many seeds
    fuzzy = {}
    rng = np.random.default_rng(42)
    tr = rng.permutation(n_all)[:20]
    Xtr = X_all[tr]
    ref_tr = rank_desc(df["Overall"].to_numpy()[tr])
    for name in ["Joint (w+cores)", "Joint + knowledge"]:
        mode, viol = methods[name]
        W = []
        for s in range(15):
            w, c = identify(Xtr, ref_tr, mode, viol, seed=100 + s)
            W.append(np.concatenate([w, c]))
        W = np.array(W)
        fuzzy[name] = {lab: [float(W[:, k].min()), float(np.median(W[:, k])),
                             float(W[:, k].max())]
                       for k, lab in enumerate([f"w_{s}" for s in SHORT] +
                                               [f"c_{s}" for s in SHORT])}
        print(f"fuzzy {name} done", flush=True)
    json.dump(fuzzy, open(f"{OUT}/fuzzy_{YEAR}.json", "w"), indent=2)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
