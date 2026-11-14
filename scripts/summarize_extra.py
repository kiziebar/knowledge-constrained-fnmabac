import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

O = sys.argv[1]


def ms(s):
    return f"{s.mean():.3f} ± {s.std():.3f}"


E1 = pd.read_csv(f"{O}/E1_cross_edition.csv")
E6 = pd.read_csv(f"{O}/E6_lp_baseline.csv")
both = pd.concat([E1[E1.method.isin(["SITW", "SITW + knowledge"])], E6])
print("=== E6 LP baseline vs SITW (cross-edition) ===")
for m in ["rw_next", "rw_next_new", "w_mae"]:
    print(f"-- {m}")
    print(both.groupby(["method", "n_train"])[m].apply(ms).unstack().to_string())
p = both.pivot_table(index=["pair", "n_train", "rep"], columns="method", values="rw_next")
for n in sorted(both.n_train.unique()):
    s = p.xs(n, level="n_train")
    d = s["SITW + knowledge"] - s["LP + knowledge"]
    pv = wilcoxon(s["SITW + knowledge"], s["LP + knowledge"]).pvalue if not np.allclose(d, 0) else np.nan
    print(f"n={n}: SITW+K vs LP+K p={pv:.2e}, median diff={np.median(d):+.4f}")

E2 = pd.read_csv(f"{O}/E2b_flag_plus_constraints.csv")
print("\n=== E2b non-monotonic: flag vs flag+constraints vs LP ===")
for scen, d in E2.groupby("scenario"):
    print(f"## {scen}")
    for m in ["rw_test", "rw_train", "w_mae", "core_err"]:
        print(f"-- {m}")
        print(d.groupby(["method", "n_train"])[m].apply(ms).unstack().to_string())
    pv = d.pivot_table(index=["n_train", "rep"], columns="method", values="rw_test")
    for n in sorted(d.n_train.unique()):
        s = pv.xs(n, level="n_train")
        print(f"n={n}: flag+constraints vs flag p={wilcoxon(s['Correct flag + constraints'], s['Correct flag']).pvalue:.2e}")

E3 = pd.read_csv(f"{O}/E3b_fuzzy_cores.csv")
print("\n=== E3b fuzzy estimates (non-monotonic) ===")
E3["ptype"] = np.where(E3.param == "core", "core", "weight")
print(E3.groupby(["scenario", "ptype", "knowledge", "n_train"])[["width", "covered", "abs_err_median"]]
        .mean().round(3).to_string())

E7 = pd.read_csv(f"{O}/E7_partial_knowledge.csv")
ref = E1[E1.method.isin(["SITW", "SITW + knowledge", "SITW + wrong knowledge"]) & E1.n_train.isin([5, 10])]
ref = ref.rename(columns={"method": "knowledge"})
allk = pd.concat([ref[["pair", "n_train", "rep", "knowledge", "rw_train", "rw_next", "w_mae"]], E7])
order = ["SITW", "ind_only", "ordinal_only", "vague", "SITW + knowledge", "mixed", "SITW + wrong knowledge"]
print("\n=== E7 partial / vague / mixed knowledge ===")
for m in ["rw_next", "rw_train", "w_mae"]:
    print(f"-- {m}")
    print(allk.groupby(["knowledge", "n_train"])[m].apply(ms).unstack().reindex(order).to_string())

print("\n=== TIME ===")
print(pd.read_csv(f"{O}/TIME.csv").round(3).to_string())
