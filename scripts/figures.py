"""Figures for the paper in the visual style of pymcdm.visuals
(default matplotlib colour cycle, dashed grid with alpha 0.5 below the data,
black-edged bars, legend above the axes in 'expand' mode, dashed lines with
markers and alpha-0.2 fills, reference values as a black dashed line with
circle markers as in pymcdm.visuals.ranking_bar).

Usage: python3 figures_pymcdm.py <results_dir> <figure_dir>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "pdf.fonttype": 42})
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
W = 12.2 / 2.54
LEGEND = dict(loc="lower left", mode="expand", borderaxespad=0.)


def pymcdm_axes(ax):
    ax.grid(linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig. 2 (E1)
E1 = pd.read_csv(f"{R}/E1_cross_edition.csv")
series = [("SITW", C[0], "*"), ("SITW + knowledge", C[1], "o"),
          ("SITW + wrong knowledge", C[2], "s")]
ns = [5, 10, 20, 40]
x = np.arange(len(ns))
fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.42))
for ax, metric, title in [(axes[0], "rw_next", "(a) all institutions of edition $t{+}1$"),
                          (axes[1], "rw_next_new", "(b) institutions new in edition $t{+}1$")]:
    g = E1.groupby(["method", "n_train"])[metric]
    for name, col, mk in series:
        m, q1, q3 = (g.mean().loc[name], g.quantile(0.25).loc[name], g.quantile(0.75).loc[name])
        ax.fill_between(x, q1.values, q3.values, color=col, alpha=0.2, linewidth=0)
        ax.plot(x, m.values, color=col, marker=mk, markersize=8 if mk == "*" else 5,
                linewidth=1, linestyle="--", label=name)
    base = E1[E1.method == "Gini"][metric].mean()
    ax.plot(x, [base] * len(x), color="k", linestyle="--", linewidth=1.5,
            marker="o", markersize=4, label="Gini weights")
    ax.set_xticks(x, labels=[str(n) for n in ns])
    ax.set_xlabel("Training decisions $n$")
    ax.set_title(title)
    pymcdm_axes(ax)
axes[0].set_ylabel("$r_w$ (mean, IQR band)")
axes[0].set_ylim(0.84, 1.005)
axes[1].set_ylim(0.3, 1.02)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.53, 0.90),
           handlelength=1.8, columnspacing=1.0, handletextpad=0.5, borderaxespad=0.)
fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.0)
fig.savefig(f"{F}/fig_e1.pdf", bbox_inches="tight")
fig.savefig(f"{F}/fig_e1.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3 (E3)
E3 = pd.read_csv(f"{R}/E3_fuzzy.csv")
d = E3[(E3.n_train == 5) & (E3.tset == 0)]
crit = ["T", "RE", "RQ", "IND", "IO"]
x = np.arange(len(crit))
width = 0.07
off = 0.13
fig, ax = plt.subplots(figsize=(W, W * 0.32))
for kn, col, o, lab in [("none", C[0], -off, "SITW"),
                        ("correct", C[1], off, "SITW + knowledge")]:
    s = d[d.knowledge == kn].set_index("criterion").loc[crit]
    ax.bar(x + o, (s.hi - s.lo).values, width, bottom=s.lo.values, color=col,
           edgecolor="black", linewidth=1, label=lab)
    ax.plot(x + o, s.med.values, linestyle="none", marker="o", markersize=5,
            color=col, markeredgecolor="black", markeredgewidth=1, zorder=3)
true = d[d.knowledge == "none"].set_index("criterion").loc[crit, "true"].values
ax.plot(x, true, linestyle="none", marker="_", markersize=16, markeredgewidth=2,
        color="k", label="True weight", zorder=4)
ax.set_xticks(x, labels=["Teaching", "Research\nenvironment", "Research\nquality",
                         "Industry", "International\noutlook"])
ax.set_ylabel("Weight value")
ax.set_ylim(0, 0.5)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
pymcdm_axes(ax)
h, l = ax.get_legend_handles_labels()
order = [1, 2, 0]
ax.legend([h[i] for i in order], [l[i] for i in order], ncol=3,
          bbox_to_anchor=(0., 1.02, 1., .102), **LEGEND)
fig.tight_layout()
fig.savefig(f"{F}/fig_e3.pdf", bbox_inches="tight")
fig.savefig(f"{F}/fig_e3.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("done")
