"""Summaries for the deployed median estimates and the knowledge-only baseline.
Usage: python3 summarize_median.py <results_dir>"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
R = sys.argv[1]
K = ["pair", "n_train", "rep"]
kn = pd.read_csv(f"{R}/knowledge_only.csv")
the = kn[kn.case == "THE"]
print("== knowledge only (THE), mean over pairs")
print(the.groupby("knowledge")[["rw_next", "rw_next_new", "w_mae"]].mean().round(4))
print(the[the.knowledge == "correct"][["pair", "rw_next", "rw_next_new"]].round(4).to_string(index=False))
gpu = kn[kn.case == "GPU"]
print("== knowledge only (GPU)")
print(gpu.groupby("n_train").rw_test.agg(["mean", "std"]).round(3))
try:
    M = pd.read_csv(f"{R}/E1_median_deploy.csv")
    E1 = pd.read_csv(f"{R}/E1_cross_edition.csv")
    LP = pd.read_csv(f"{R}/E6_lp_baseline.csv")
    print("== E1 median deployment (n_tasks per n):", M.groupby("n_train").rep.count().to_dict())
    for col in ["rw_next", "rw_next_new", "w_mae", "rw_train"]:
        print(col); print(M.groupby(["method", "n_train"])[col].agg(["mean", "std"]).round(3).unstack(0))
    print("per pair rw_next"); print(M.groupby(["method", "n_train", "pair"]).rw_next.mean().unstack().round(3))
    kno = the[the.knowledge == "correct"].set_index("pair").rw_next
    a = M[M.method == "SITW + knowledge (median)"].copy()
    a["kn_only"] = a.pair.map(kno)
    for n in sorted(M.n_train.unique()):
        x = a[a.n_train == n]
        p = wilcoxon(x.rw_next, x.kn_only).pvalue
        print(f"n={n} median+K {x.rw_next.mean():.4f} vs kn-only {x.kn_only.mean():.4f} p={p:.2g} share>={(x.rw_next>=x.kn_only).mean():.2f}")
    s = E1[E1.method == "SITW + knowledge"].set_index(K).rw_next
    m = M[M.method == "SITW + knowledge (median)"].set_index(K).rw_next
    s0 = E1[E1.method == "SITW"].set_index(K).rw_next
    m0 = M[M.method == "SITW (median)"].set_index(K).rw_next
    for n in sorted(M.n_train.unique()):
        j = m.xs(n, level=1).index.intersection(s.xs(n, level=1).index)
        mm, ss = m.xs(n, level=1).loc[j], s.xs(n, level=1).loc[j]
        mm0 = m0.xs(n, level=1).loc[j]
        print(f"n={n}: single {ss.mean():.4f} median {mm.mean():.4f} p={wilcoxon(mm, ss).pvalue:.2g}; "
              f"median K vs median none p={wilcoxon(mm, mm0).pvalue:.2g}")
    print(LP.columns.tolist())
except FileNotFoundError as ex:
    print("missing", ex)
try:
    G = pd.read_csv(f"{R}/E6_median_deploy.csv")
    print("== GPU median deployment")
    print(G.groupby(["method", "n_train"]).rw_test.agg(["mean", "std"]).round(3).unstack(0).T)
    print(G.groupby(["method", "n_train"]).rw_train.mean().round(3).unstack())
    g = gpu.set_index(["n_train", "rep"]).rw_test
    for meth in G.method.unique():
        for n in [10, 15, 20]:
            x = G[(G.method == meth) & (G.n_train == n)].set_index(["n_train", "rep"]).rw_test
            print(meth, n, round(x.mean(), 3), "vs kn-only", round(g.loc[x.index].mean(), 3),
                  "p=%.2g" % wilcoxon(x, g.loc[x.index]).pvalue)
except FileNotFoundError as ex:
    print("missing", ex)
