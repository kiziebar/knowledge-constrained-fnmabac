import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

O = sys.argv[1]
S = pd.read_csv(f"{O}/E8_gpu_split.csv")
order = ["Gini (FN-MABAC)", "STFN-MABAC (Gini w, all cores)", "LP + knowledge", "SITW",
         "SITW + knowledge", "SITW + wrong knowledge", "Flag C3", "Flag C3 + knowledge",
         "Wrong flag C4 + knowledge", "All cores + weights"]


def ms(s):
    return f"{s.mean():.3f} ± {s.std():.3f}"


for m in ["rw_test", "rw_train", "core_C3"]:
    print(f"\n-- {m}")
    print(S.groupby(["method", "n_train"])[m].apply(ms).unstack().reindex(order).to_string())
p = S.pivot_table(index=["n_train", "rep"], columns="method", values="rw_test")
for n in sorted(S.n_train.unique()):
    s = p.xs(n, level="n_train")
    for a, b in [("Flag C3 + knowledge", "SITW + knowledge"), ("Flag C3 + knowledge", "Flag C3"),
                 ("Flag C3 + knowledge", "All cores + weights"), ("Flag C3 + knowledge", "Gini (FN-MABAC)"),
                 ("Flag C3 + knowledge", "STFN-MABAC (Gini w, all cores)"), ("SITW + knowledge", "SITW")]:
        print(f"n={n}: {a} vs {b}: p={wilcoxon(s[a], s[b]).pvalue:.1e}, mean diff={np.mean(s[a]-s[b]):+.3f}")

F = pd.read_csv(f"{O}/E8_gpu_full.csv")
print("\n-- full data (30 cards), 15 seeds: train rw and fuzzy C3 core / key weights")
g = F.groupby("method")
out = pd.DataFrame({
    "rw_train": g.rw_train.apply(ms),
    "C3 core (min,med,max)": g.c_C3.apply(lambda s: f"({s.min():.0f}, {s.median():.0f}, {s.max():.0f})"),
    "w_C1 (min,med,max)": g.w_C1.apply(lambda s: f"({s.min():.3f}, {s.median():.3f}, {s.max():.3f})"),
    "w_C6 (min,med,max)": g.w_C6.apply(lambda s: f"({s.min():.3f}, {s.median():.3f}, {s.max():.3f})"),
})
print(out.reindex([o for o in order if o in out.index]).to_string())
print("\nmedian weights per method:")
print(g[[f"w_C{j}" for j in range(1, 9)]].median().round(3).to_string())
print("\nweight TFN widths (mean over criteria):")
print(g[[f"w_C{j}" for j in range(1, 9)]].apply(lambda d: (d.max() - d.min()).mean()).round(3).to_string())
