"""Supplementary experiments.

E2b  correct flag + weight constraints vs correct flag only (non-monotonic scenarios)
E3b  fuzzy estimates of learned cores (and weights) in the non-monotonic scenarios
E6   LP preference-disaggregation baseline (fixed marginals), with/without knowledge,
     in the cross-edition protocol (E1) and in the non-monotonic scenarios (E2)
E7   partial / vague / mixed expert knowledge in the cross-edition protocol
TIME runtime of one identification

Usage: python3 -I extra_experiments.py <E2b|E3b|E6|E7|TIME> <csv_dir> <out_dir> <workers>
"""
import os
import sys
WHICH, CSV, OUT, WORKERS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["x", "NONE", CSV, OUT, str(WORKERS)]
import full_experiments as F          # noqa: E402  (reuses identify, data, KN, SCEN)

import time                             # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from scipy.optimize import linprog      # noqa: E402

e, M, HI = F.e, F.M, F.HI
REPS = 30


# ------------------------------------------------------------ extra knowledge sets
def kn_ind_only(w):            # partial: only "Industry is the smallest weight"
    T, RE, RQ, IND, IO = w
    return sum(max(0.0, IND - x) for x in (T, RE, RQ, IO))


def kn_ordinal_only(w):        # partial: ordinal relations, no bounds
    T, RE, RQ, IND, IO = w
    v = sum(max(0.0, IND - x) for x in (T, RE, RQ, IO))
    v += sum(max(0.0, IO - x) for x in (T, RE, RQ))
    return v


def kn_vague(w):               # vague: loose bounds (>= 0.1 instead of 0.2)
    T, RE, RQ, IND, IO = w
    v = kn_ordinal_only(w)
    v += sum(max(0.0, 0.1 - x) for x in (T, RE, RQ))
    return v


def kn_mixed(w):               # mixed: correct ordinal relations + one wrong bound
    T, RE, RQ, IND, IO = w
    return kn_ordinal_only(w) + max(0.0, RQ - 0.1)


F.KN.update({"ind_only": kn_ind_only, "ordinal_only": kn_ordinal_only,
             "vague": kn_vague, "mixed": kn_mixed})


# ------------------------------------------------------------ LP baseline
def lp_weights(X, ref_rank, knowledge=False, C=10.0):
    """Preference disaggregation with fixed linear marginals mu_j(x) = x/100:
    maximise margin eps minus C * sum of slacks over all ranked pairs."""
    N = np.clip(X / 100.0, 0, 1)
    order = np.argsort(ref_rank)                  # best first
    pairs = [(order[i], order[k]) for i in range(len(order)) for k in range(i + 1, len(order))]
    P = len(pairs)
    # variables: w (M), eps (1), sigma (P)
    nv = M + 1 + P
    c = np.zeros(nv); c[M] = -1.0; c[M + 1:] = C
    A, b = [], []
    for p, (a, z) in enumerate(pairs):            # -(w.(Na-Nz)) + eps - sigma <= 0
        row = np.zeros(nv); row[:M] = -(N[a] - N[z]); row[M] = 1.0; row[M + 1 + p] = -1.0
        A.append(row); b.append(0.0)
    if knowledge:                                 # same statements as kn_correct (linear)
        T, RE, RQ, IND, IO = range(5)
        for x in (T, RE, RQ, IO):                 # IND <= x
            row = np.zeros(nv); row[IND] = 1; row[x] = -1; A.append(row); b.append(0.0)
        for x in (T, RE, RQ):                     # IO <= x
            row = np.zeros(nv); row[IO] = 1; row[x] = -1; A.append(row); b.append(0.0)
        for x in (T, RE, RQ):                     # x >= 0.2
            row = np.zeros(nv); row[x] = -1; A.append(row); b.append(-0.2)
    Aeq = np.zeros((1, nv)); Aeq[0, :M] = 1.0
    bounds = [(0, 1)] * M + [(None, 1.0)] + [(0, None)] * P
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    if not res.success:
        return np.full(M, 1 / M)
    w = np.clip(res.x[:M], 0, None)
    return w / w.sum()


# ------------------------------------------------------------ tasks
def e2b_task(args):
    scen, n, r = args
    w_syn, c_syn, j = F.SCEN[scen]
    d, X, _ = F.data(2025)
    d2, X2, _ = F.data(2026)
    pref_tr, pref_te = e.fn_mabac(X, w_syn, c_syn), e.fn_mabac(X2, w_syn, c_syn)
    idx = np.random.default_rng(50_000 + 100 * n + r).permutation(len(d))[:n]
    ref_tr, ref_te = e.rank_desc(pref_tr[idx]), e.rank_desc(pref_te)
    flag = [False] * 5; flag[j] = True
    out = []
    for name, kn in [("Correct flag", "none"), ("Correct flag + constraints", "correct")]:
        w, c = F.identify(X[idx], ref_tr, flag, kn, seed=r)
        out.append(dict(scenario=scen, n_train=n, rep=r, method=name,
                        rw_train=e.rw(e.fn_mabac(X[idx], w, c), ref_tr),
                        rw_test=e.rw(e.fn_mabac(X2, w, c), ref_te),
                        w_mae=np.abs(w - w_syn).mean(), core_learned=c[j],
                        core_err=abs(c[j] - c_syn[j])))
    for name, kn in [("LP", False), ("LP + knowledge", True)]:
        w = lp_weights(X[idx], ref_tr, kn)
        c = np.full(M, HI)
        out.append(dict(scenario=scen, n_train=n, rep=r, method=name,
                        rw_train=e.rw(e.fn_mabac(X[idx], w, c), ref_tr),
                        rw_test=e.rw(e.fn_mabac(X2, w, c), ref_te),
                        w_mae=np.abs(w - w_syn).mean(), core_learned=np.nan,
                        core_err=np.nan))
    return out


def e3b_task(args):
    scen, n, tset, kn = args
    w_syn, c_syn, j = F.SCEN[scen]
    d, X, _ = F.data(2025)
    pref = e.fn_mabac(X, w_syn, c_syn)
    idx = np.random.default_rng(80_000 + 100 * n + tset).permutation(len(d))[:n]
    ref = e.rank_desc(pref[idx])
    flag = [False] * 5; flag[j] = True
    S = []
    for s in range(15):
        w, c = F.identify(X[idx], ref, flag, kn, seed=2000 + s)
        S.append(np.concatenate([w, [c[j]]]))
    S = np.array(S)
    names = [f"w_{x}" for x in e.SHORT] + ["core"]
    truth = np.concatenate([w_syn, [c_syn[j]]])
    out = []
    for k, nm in enumerate(names):
        lo, med, hi = S[:, k].min(), np.median(S[:, k]), S[:, k].max()
        out.append(dict(scenario=scen, n_train=n, tset=tset, knowledge=kn, param=nm,
                        lo=lo, med=med, hi=hi, width=hi - lo, true=truth[k],
                        covered=bool(lo - 1e-9 <= truth[k] <= hi + 1e-9),
                        abs_err_median=abs(med - truth[k])))
    return out


def e6_task(args):
    (y_tr, y_te), n, r = args
    dtr, Xtr_all, otr = F.data(y_tr)
    dte, Xte, ote = F.data(y_te)
    new = ~dte.Name.isin(set(dtr.Name)).to_numpy()
    idx = np.random.default_rng(10_000 * y_tr + 100 * n + r).permutation(len(dtr))[:n]
    Xs, rs = Xtr_all[idx], e.rank_desc(otr[idx])
    out = []
    for name, kn in [("LP", False), ("LP + knowledge", True)]:
        w = lp_weights(Xs, rs, kn); c = np.full(M, HI)
        out.append(dict(pair=f"{y_tr}->{y_te}", n_train=n, rep=r, method=name,
                        rw_train=e.rw(e.fn_mabac(Xs, w, c), rs),
                        rw_next=e.rw(e.fn_mabac(Xte, w, c), e.rank_desc(ote)),
                        rw_next_new=e.rw(e.fn_mabac(Xte[new], w, c), e.rank_desc(ote[new])),
                        w_mae=np.abs(w - e.W_TRUE).mean()))
    return out


def e7_task(args):
    (y_tr, y_te), n, r, kn = args
    dtr, Xtr_all, otr = F.data(y_tr)
    dte, Xte, ote = F.data(y_te)
    idx = np.random.default_rng(10_000 * y_tr + 100 * n + r).permutation(len(dtr))[:n]
    Xs, rs = Xtr_all[idx], e.rank_desc(otr[idx])
    w, c = F.identify(Xs, rs, F.NONE, kn, seed=r)
    return [dict(pair=f"{y_tr}->{y_te}", n_train=n, rep=r, knowledge=kn,
                 rw_train=e.rw(e.fn_mabac(Xs, w, c), rs),
                 rw_next=e.rw(e.fn_mabac(Xte, w, c), e.rank_desc(ote)),
                 w_mae=np.abs(w - e.W_TRUE).mean())]


def timing():
    d, X, o = F.data(2026)
    rows = []
    for n in [5, 40, 200]:
        for mode, mask in [("weights", F.NONE), ("weights+1 core", [False] * 4 + [True]),
                           ("weights+5 cores", F.ALL)]:
            idx = np.random.default_rng(n).permutation(len(d))[:n]
            ts = []
            for s in range(5):
                t = time.perf_counter()
                F.identify(X[idx], e.rank_desc(o[idx]), mask, "correct", seed=s)
                ts.append(time.perf_counter() - t)
            rows.append(dict(n_train=n, params=mode, secs_mean=np.mean(ts), secs_sd=np.std(ts)))
        if n > 40:
            continue
        t = time.perf_counter()
        for s in range(20):
            lp_weights(X[idx], e.rank_desc(o[idx]), True)
        rows.append(dict(n_train=n, params="LP + knowledge", secs_mean=(time.perf_counter() - t) / 20,
                         secs_sd=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/TIME.csv", index=False)
    print(df.round(3).to_string(), flush=True)


if __name__ == "__main__":
    if WHICH == "E2b":
        F.run("E2b_flag_plus_constraints", e2b_task,
              [(s, n, r) for s in F.SCEN for n in [10, 20, 40] for r in range(REPS)])
    elif WHICH == "E3b":
        F.run("E3b_fuzzy_cores", e3b_task,
              [(s, n, t, kn) for s in F.SCEN for n in [10, 20, 40] for t in range(5)
               for kn in ["none", "correct"]])
    elif WHICH == "E6":
        F.run("E6_lp_baseline", e6_task,
              [(p, n, r) for p in F.PAIRS for n in F.NS for r in range(REPS)])
    elif WHICH == "E7":
        F.run("E7_partial_knowledge", e7_task,
              [(p, n, r, kn) for p in F.PAIRS for n in [5, 10] for r in range(REPS)
               for kn in ["ind_only", "ordinal_only", "vague", "mixed"]])
    elif WHICH == "TIME":
        timing()
        print("TIME DONE", flush=True)
