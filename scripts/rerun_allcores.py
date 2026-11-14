"""Re-run the variants that learn all reference points with JADE (300 epochs, pop 50),
because PSO (150-200 epochs) did not converge in the enlarged parameter space.

Usage: python3 -I rerun_allcores.py <E1|E2|GPU_SPLIT|GPU_FULL> <workers>
"""
import os
import sys
WHICH, WORKERS = sys.argv[1], int(sys.argv[2])
CSV, OUT = "../data/THE-World-University-Rankings/outputs/csv", "../results"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["x", "NONE", CSV, OUT, str(WORKERS)]
import full_experiments as F            # noqa: E402
import numpy as np                       # noqa: E402
from scipy.stats import rankdata         # noqa: E402

e, M, HI = F.e, F.M, F.HI


def e1_task(args):
    (y_tr, y_te), n, r = args
    dtr, Xtr_all, otr = F.data(y_tr)
    dte, Xte, ote = F.data(y_te)
    new = ~dte.Name.isin(set(dtr.Name)).to_numpy()
    idx = np.random.default_rng(10_000 * y_tr + 100 * n + r).permutation(len(dtr))[:n]
    Xs, rs = Xtr_all[idx], e.rank_desc(otr[idx])
    out = []
    for name, (mask, kn, floor) in [("All cores free", (F.ALL, "none", None)),
                                    ("All cores + knowledge", (F.ALL, "correct", 90.0))]:
        w, c = F.identify(Xs, rs, mask, kn, seed=r, core_floor=floor, opt="jade")
        out.append(dict(pair=f"{y_tr}->{y_te}", n_train=n, rep=r, method=name,
                        rw_train=e.rw(e.fn_mabac(Xs, w, c), rs),
                        rw_next=e.rw(e.fn_mabac(Xte, w, c), e.rank_desc(ote)),
                        rw_next_new=e.rw(e.fn_mabac(Xte[new], w, c), e.rank_desc(ote[new])),
                        n_new=int(new.sum()), w_mae=np.abs(w - e.W_TRUE).mean(),
                        core_mae=np.abs(c - e.CORE_TRUE).mean()))
    return out


def e2_task(args):
    scen, n, r = args
    w_syn, c_syn, j = F.SCEN[scen]
    d, X, _ = F.data(2025)
    d2, X2, _ = F.data(2026)
    pref_tr, pref_te = e.fn_mabac(X, w_syn, c_syn), e.fn_mabac(X2, w_syn, c_syn)
    idx = np.random.default_rng(50_000 + 100 * n + r).permutation(len(d))[:n]
    ref_tr, ref_te = e.rank_desc(pref_tr[idx]), e.rank_desc(pref_te)
    out = []
    for name, floor in [("All cores free", None), ("All cores, monotonic belief", 90.0)]:
        w, c = F.identify(X[idx], ref_tr, F.ALL, "none", seed=r, core_floor=floor, opt="jade")
        out.append(dict(scenario=scen, n_train=n, rep=r, method=name,
                        rw_train=e.rw(e.fn_mabac(X[idx], w, c), ref_tr),
                        rw_test=e.rw(e.fn_mabac(X2, w, c), ref_te),
                        w_mae=np.abs(w - w_syn).mean(), core_err=abs(c[j] - c_syn[j]),
                        core_learned=c[j]))
    return out


def gpu_split_task(args):
    sys.argv = ["x", "NONE", "/tmp", "1"]
    import gpu_experiment as GE
    G = GE.G
    n, r = args
    idx = np.random.default_rng(60_000 + 100 * n + r).permutation(30)
    tr, te = idx[:n], idx[n:]
    ref_tr, ref_te = rankdata(G.REF[tr]), rankdata(G.REF[te])
    out = []
    for name, fw in [("All cores + weights", None), ("STFN-MABAC (Gini w, all cores)", G.W_GINI)]:
        w, c = GE.identify(G.X[tr], ref_tr, GE.ALL, "none", seed=r, fixed_w=fw, opt="jade")
        out.append(GE.evaluate(name, w, c, tr, te, ref_tr, ref_te, n, r))
    return out


def gpu_full_task(args):
    sys.argv = ["x", "NONE", "/tmp", "1"]
    import gpu_experiment as GE
    G = GE.G
    name, s = args
    fw = G.W_GINI if name.startswith("STFN") else None
    w, c = GE.identify(G.X, G.REF, GE.ALL, "none", seed=3000 + s, fixed_w=fw, opt="jade")
    return [dict(method=name, seed=s, rw_train=GE.rw(GE.fn_mabac(G.X, w, c), G.REF),
                 **{f"w_{x}": w[k] for k, x in enumerate(G.SHORT)},
                 **{f"c_{x}": c[k] for k, x in enumerate(G.SHORT)})]


if __name__ == "__main__":
    if WHICH == "E1":
        F.run("E1_allcores_jade", e1_task, [(p, n, r) for p in F.PAIRS for n in F.NS for r in range(30)])
    elif WHICH == "E2":
        F.run("E2_allcores_jade", e2_task, [(s, n, r) for s in F.SCEN for n in [10, 20, 40] for r in range(30)])
    elif WHICH == "GPU_SPLIT":
        F.run("E8_allcores_jade_split", gpu_split_task, [(n, r) for n in [10, 15, 20] for r in range(30)])
    elif WHICH == "GPU_FULL":
        F.run("E8_allcores_jade_full", gpu_full_task,
              [(nm, s) for nm in ["All cores + weights", "STFN-MABAC (Gini w, all cores)"] for s in range(15)])
