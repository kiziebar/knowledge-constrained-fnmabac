"""Summaries + statistical tests for the full experiment suite."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

O = sys.argv[1]


def ms(s):
    return f"{s.mean():.3f} ± {s.std():.3f}"


def wil(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return float("nan")
    return wilcoxon(a, b).pvalue


# ---------------- E1
E1 = pd.read_csv(f"{O}/E1_cross_edition.csv")
ORDER = ["Equal", "Entropy", "CRITIC", "Gini", "SITW", "SITW + knowledge",
         "SITW + wrong knowledge", "All cores free", "All cores + knowledge"]
for metric in ["rw_next", "rw_next_new", "w_mae"]:
    print(f"\n=== E1 {metric}: mean ± sd over pairs×reps ===")
    t = E1.groupby(["method", "n_train"])[metric].apply(ms).unstack().reindex(ORDER)
    print(t.to_string())
print("\n=== E1 rw_next by pair (n=5) ===")
print(E1[E1.n_train == 5].groupby(["method", "pair"])["rw_next"].mean()
      .unstack().reindex(ORDER).round(3).to_string())
print("\n=== E1 Wilcoxon: SITW + knowledge vs SITW (paired by pair,n,rep) ===")
p = E1.pivot_table(index=["pair", "n_train", "rep"], columns="method", values="rw_next")
for n in sorted(E1.n_train.unique()):
    s = p.xs(n, level="n_train")
    print(f"n={n}: rw_next p={wil(s['SITW + knowledge'], s['SITW']):.2e}, "
          f"median diff={np.median(s['SITW + knowledge'] - s['SITW']):+.4f}")
pn = E1.pivot_table(index=["pair", "n_train", "rep"], columns="method", values="rw_next_new")
for n in sorted(E1.n_train.unique()):
    s = pn.xs(n, level="n_train")
    print(f"n={n}: rw_next_new p={wil(s['SITW + knowledge'], s['SITW']):.2e}")
print("\n=== E1 conflict signal: train fit, correct vs wrong knowledge ===")
pt = E1.pivot_table(index=["pair", "n_train", "rep"], columns="method", values="rw_train")
for n in sorted(E1.n_train.unique()):
    s = pt.xs(n, level="n_train")
    c, w = s["SITW + knowledge"], s["SITW + wrong knowledge"]
    auc = (np.mean(c.to_numpy()[:, None] > w.to_numpy()[None, :]) +
           0.5 * np.mean(c.to_numpy()[:, None] == w.to_numpy()[None, :]))
    print(f"n={n}: train rw correct {ms(c)}, wrong {ms(w)}, AUC={auc:.3f}")

# ---------------- E2
E2 = pd.read_csv(f"{O}/E2_nonmonotonic.csv")
O2 = ["No flag", "Correct flag", "Wrong flag", "All cores free", "All cores, monotonic belief"]
for scen, d in E2.groupby("scenario"):
    print(f"\n=== E2 {scen} ===")
    for metric in ["rw_train", "rw_test", "core_learned", "w_mae"]:
        print(f"-- {metric}")
        print(d.groupby(["method", "n_train"])[metric].apply(ms).unstack().reindex(O2).to_string())

# ---------------- E3
E3 = pd.read_csv(f"{O}/E3_fuzzy.csv")
print("\n=== E3 fuzzy estimates: mean width / coverage / median abs err ===")
print(E3.groupby(["knowledge", "n_train"])[["width", "covered", "abs_err_median"]]
        .mean().round(3).to_string())
print(E3.groupby(["knowledge", "criterion"])[["width", "covered"]].mean().round(3).to_string())

# ---------------- E4
E4 = pd.read_csv(f"{O}/E4_lambda.csv")
print("\n=== E4 lambda sensitivity ===")
for metric in ["rw_next", "rw_train"]:
    print(f"-- {metric}")
    print(E4.groupby(["knowledge", "n_train", "lam"])[metric].apply(ms).unstack().to_string())

# ---------------- E5
E5 = pd.read_csv(f"{O}/E5_rank_reversal.csv")
print("\n=== E5 rank reversal ===")
print(E5.groupby(["scenario", "method"])[["pair_flip_rate", "any_reversal"]].mean().round(4).to_string())
