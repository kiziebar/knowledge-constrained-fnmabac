"""Deployed point estimate = median of S = 15 identification runs (Section 4.4),
plus the knowledge-only baseline (centre of the region admitted by the expert
knowledge, estimated from uniformly sampled weight vectors).

Usage:
  python3 -I median_deploy.py E1  <csv_dir> <out_dir> <workers>
  python3 -I median_deploy.py GPU <csv_dir> <out_dir> <workers>
  python3 -I median_deploy.py KNONLY <csv_dir> <out_dir> 1
"""
import sys
WHICH, CSV, OUT, WORKERS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.argv = ["x", "NONE", CSV, OUT, str(WORKERS)]
import warnings                                   # noqa: E402
warnings.filterwarnings("ignore")
from multiprocessing import Pool                  # noqa: E402
import numpy as np                                # noqa: E402
import pandas as pd                               # noqa: E402
from scipy.stats import rankdata                  # noqa: E402
import full_experiments as F                      # noqa: E402
import extra_experiments                          # noqa: E402,F401  registers partial knowledge sets
import experiment as e                            # noqa: E402

S = 15
E1_VARIANTS = ["SITW", "SITW + knowledge"]
GPU_VARIANTS = ["SITW", "SITW + knowledge", "Flag C3", "Flag C3 + knowledge"]


def median_of_runs(results):
    W = np.array([w for w, _ in results]); C = np.array([c for _, c in results])
    w = np.median(W, 0); w = w / w.sum()
    return w, np.median(C, 0)


def e1_task(args):
    (y_tr, y_te), n, r = args
    dtr, Xtr_all, otr = F.data(y_tr)
    dte, Xte, ote = F.data(y_te)
    ref_te = e.rank_desc(ote)
    new = ~dte.Name.isin(set(dtr.Name)).to_numpy()
    ref_new = e.rank_desc(ote[new])
    idx = np.random.default_rng(10_000 * y_tr + 100 * n + r).permutation(len(dtr))[:n]
    Xs, rs = Xtr_all[idx], e.rank_desc(otr[idx])
    out = []
    for name in E1_VARIANTS:
        mask, kn, floor = F.E1_IDENT[name]
        runs = [F.identify(Xs, rs, mask, kn, seed=100_000 + 100 * r + s, core_floor=floor)
                for s in range(S)]
        w, c = median_of_runs(runs)
        out.append(dict(pair=f"{y_tr}->{y_te}", n_train=n, rep=r, method=name + " (median)",
                        rw_train=e.rw(e.fn_mabac(Xs, w, c), rs),
                        rw_train_runs=np.mean([e.rw(e.fn_mabac(Xs, *x), rs) for x in runs]),
                        rw_next=e.rw(e.fn_mabac(Xte, w, c), ref_te),
                        rw_next_new=e.rw(e.fn_mabac(Xte[new], w, c), ref_new),
                        w_mae=np.abs(w - e.W_TRUE).mean(),
                        **{f"w_{k}": w[i] for i, k in enumerate(e.SHORT)}))
    return out


def gpu_task(args):
    import gpu_experiment as g
    import gpu_data as G
    n, r = args
    idx = np.random.default_rng(60_000 + 100 * n + r).permutation(30)
    tr, te = idx[:n], idx[n:]
    ref_tr = rankdata(G.REF[tr]); ref_te = rankdata(G.REF[te])
    out = []
    for name in GPU_VARIANTS:
        mask, kn, fw = g.CONFIGS[name]
        runs = [g.identify(G.X[tr], ref_tr, mask, kn, seed=100_000 + 100 * r + s) for s in range(S)]
        w, c = median_of_runs(runs)
        d = g.evaluate(name + " (median)", w, c, tr, te, ref_tr, ref_te, n, r)
        d["rw_train_runs"] = np.mean([g.rw(g.fn_mabac(G.X[tr], *x), ref_tr) for x in runs])
        out.append(d)
    return out


def central_weights(viol, m, samples=400_000, seed=0):
    W = np.random.default_rng(seed).dirichlet(np.ones(m), samples)
    ok = W[np.array([viol(w) == 0 for w in W])]
    c = ok.mean(0)
    return c / c.sum(), len(ok)


def knowledge_only():
    sys.argv = ["x", "NONE", OUT, "1"]
    import gpu_experiment as g
    import gpu_data as G
    rows = []
    for kn in ["correct", "ordinal_only", "vague", "ind_only"]:
        viol = F.KN[kn]
        w, k = central_weights(viol, 5)
        for (y_tr, y_te) in F.PAIRS:
            dtr, _, _ = F.data(y_tr)
            dte, Xte, ote = F.data(y_te)
            new = ~dte.Name.isin(set(dtr.Name)).to_numpy()
            c = np.full(5, 100.0)
            rows.append(dict(case="THE", knowledge=kn, pair=f"{y_tr}->{y_te}", n_feasible=k,
                             rw_next=e.rw(e.fn_mabac(Xte, w, c), e.rank_desc(ote)),
                             rw_next_new=e.rw(e.fn_mabac(Xte[new], w, c), e.rank_desc(ote[new])),
                             w_mae=np.abs(w - e.W_TRUE).mean(),
                             **{f"w_{s}": w[i] for i, s in enumerate(e.SHORT)}))
    w, k = central_weights(g.kn_correct, 8)
    for n in [10, 15, 20]:
        for r in range(30):
            idx = np.random.default_rng(60_000 + 100 * n + r).permutation(30)
            te = idx[n:]
            rows.append(dict(case="GPU", knowledge="correct", n_train=n, rep=r, n_feasible=k,
                             rw_test=g.rw(g.fn_mabac(G.X[te], w, G.CORE_TYPE), rankdata(G.REF[te])),
                             **{f"w_{s}": w[i] for i, s in enumerate(G.SHORT)}))
    pd.DataFrame(rows).to_csv(f"{OUT}/knowledge_only.csv", index=False)
    print("knowledge_only DONE")


def run_resumable(name, fn, tasks, keycols):
    """Append each finished task to the CSV immediately and skip finished tasks,
    so that an interrupted run can be resumed."""
    import os
    path = f"{OUT}/{name}.csv"
    done = set()
    if os.path.exists(path):
        old = pd.read_csv(path)
        done = set(map(tuple, old[keycols].astype(str).values))
    todo = [t for t in tasks if tuple(map(str, t[1])) not in done]
    print(f"{name}: {len(tasks) - len(todo)} done, {len(todo)} to do", flush=True)
    with Pool(WORKERS) as pool:
        for i, rows in enumerate(pool.imap_unordered(fn, [t[0] for t in todo], chunksize=1), 1):
            df = pd.DataFrame(rows)
            df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            if i % 10 == 0:
                print(f"{name}: {i}/{len(todo)}", flush=True)
    print(f"{name} DONE", flush=True)


if __name__ == "__main__":
    if WHICH == "E1":
        tasks = [((p, n, r), (f"{p[0]}->{p[1]}", n, r)) for r in range(F.REPS) for n in F.NS for p in F.PAIRS]
        run_resumable("E1_median_deploy", e1_task, tasks, ["pair", "n_train", "rep"])
    elif WHICH == "GPU":
        sys.argv = ["x", "NONE", OUT, "1"]
        import gpu_experiment  # noqa: F401
        tasks = [((n, r), (n, r)) for r in range(30) for n in [10, 15, 20]]
        run_resumable("E6_median_deploy", gpu_task, tasks, ["n_train", "rep"])
    elif WHICH == "KNONLY":
        knowledge_only()
