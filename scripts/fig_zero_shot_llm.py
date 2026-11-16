"""Figures for the zero-shot LLM trial (docs/results_zero_shot.md, section 3). Reads the outputs of
scripts/zero_shot_study.py; makes no API call.

    python scripts/fig_zero_shot_llm.py

Writes reports/library/figs/zero_shot_llm_examples.png (one typical cell per dataset) and
zero_shot_llm_errors.png (error by method, per dataset and by horizon).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee_analogs import zero_shot as z  # noqa: E402

OUT = ROOT / "reports" / "library"
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
NAMES = {"lsd1": "LSD", "mohtat": "Mohtat (NMC)", "severson": "Severson (LFP)", "stanford_dyn": "Stanford EV (NCA)",
         "stanford_matrix": "Stanford matrix (NCA)", "tongji": "Tongji (NCA/NCM)"}
# method: (label, colour, line style)
M = {"llm_B": ("Opus B: + chemistry, mechanisms", "#882255", "-"),
     "llm_C": ("Opus C: no analogs", "#CC6677", (0, (3, 1))),
     "llm_A": ("Opus A: curve + analogs", "#AA4499", ":"),
     "haiku_B": ("Haiku B: + chemistry, mechanisms", "#882255", (0, (1, 1))),
     "haiku_C": ("Haiku C: no analogs", "#CC6677", (0, (1, 1))),
     "haiku_A": ("Haiku A: curve + analogs", "#AA4499", (0, (1, 1))),
     "ensemble_llm": ("mean of 3 + LLM B", "#000000", "-"),
     "ensemble": ("mean of analogs, Chronos-2, trend", "#555555", "--"),
     "lib_shape": ("library analogs", "#E69F00", "-"),
     "chronos2": ("Chronos-2", "#56B4E9", "-."),
     "own_linear": ("own linear trend", "#0072B2", "--"),
     "within_shape": ("same-dataset analogs (upper bound)", "#009E73", "-")}


def fig_examples(F, curves, path):
    dss = list(NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(180 * mm, 105 * mm), layout="constrained")
    for ax, ds in zip(axes.flat, dss):
        D = F[F["dataset"] == ds]
        C = D.dropna(subset=["llm_B", "own_linear", "chronos2", "lib_shape"])
        e = (C["llm_B"] - C["actual"]).abs().groupby(C["id"]).mean().sort_values() * 100
        q = e.index[len(e) // 2]  # median-error cell for LLM B: a typical case, not a showcase
        g = D[D["id"] == q].sort_values("efc")
        t, s = curves[q]
        t90 = g["t90"].iloc[0]
        seen = t <= t90 * 1.0001
        fut = (t > t90) & (t <= 3 * t90)
        ax.axvspan(0, t90, color="#000000", alpha=0.04, lw=0)
        ax.plot(t[seen], s[seen] * 100, ".", color="k", ms=2, label="seen (to 90 %)")
        ax.plot(t[fut], s[fut] * 100, ".", color="#a0a0a0", ms=2, label="actual future")
        ax.fill_between(g["efc"], g["llm_B_lo"] * 100, g["llm_B_hi"] * 100, color="#882255", alpha=0.15, lw=0,
                        label="LLM B 10-90 % band")
        for m in ("llm_B", "lib_shape", "chronos2", "own_linear"):
            lab, c, ls = M[m]
            gm = g.dropna(subset=[m])
            ax.plot(np.r_[t90, gm["efc"]], np.r_[90, gm[m] * 100], color=c, ls=ls, lw=1.0, label=lab)
        ax.axhline(90, color="k", lw=0.3, ls=":")
        ax.set_title(f"{NAMES[ds]}  (LLM B error {e[q]:.1f})", fontsize=6.5)
        ax.set_xlim(0, max(t[fut].max(), g["efc"].max()) * 1.05)
        lo = min(np.nanmin(s[fut]) if fut.any() else 0.8, g[["llm_B_lo", "own_linear"]].min().min()) * 100
        ax.set_ylim(max(lo - 2, 40), 101)
        ax.tick_params(labelsize=5.5)
        ax.set_xlabel("equivalent full cycles", fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH (%)", fontsize=6)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=5.5, frameon=False)
    fig.suptitle("One typical cell per dataset (median LLM B error). Shaded: the part the forecasters see.",
                 fontsize=7, x=0.01, ha="left")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_errors(path):
    O = pd.read_csv(OUT / "zero_shot_overall.csv", index_col=0)
    P = pd.read_csv(OUT / "zero_shot_by_dataset.csv")
    order = [m for m in ("within_shape", "ensemble_llm", "ensemble", "llm_B", "llm_C", "llm_A", "haiku_B",
                         "haiku_C", "haiku_A", "lib_shape", "chronos2", "own_linear") if m in O.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(180 * mm, 82 * mm), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(order))[::-1]
    a1.barh(y, O.loc[order, "MAE"], color=[M[m][1] for m in order], alpha=0.35, height=0.7)
    marks = dict(zip(NAMES, "osDv^P"))
    for ds, mk in marks.items():
        v = P[P["dataset"] == ds].set_index("method")["MAE"].reindex(order)
        a1.plot(v, y + np.linspace(-0.25, 0.25, 6)[list(NAMES).index(ds)], mk, ms=2.8, mfc="none", mew=0.5,
                color="#333333", label=NAMES[ds], ls="none")
    for yi, m in zip(y, order):
        a1.text(0.05, yi, f"{O.loc[m, 'MAE']:.2f}", va="center", fontsize=5.5, weight="bold")
    a1.set_yticks(y, [M[m][0] for m in order], fontsize=5.5)
    a1.set_xlabel("error over the future to 3 × t90 (SOH points)", fontsize=6)
    a1.set_title("Bars: mean over the 6 datasets; markers: each dataset", fontsize=6.5)
    a1.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    a1.set_xlim(0, 3.8)
    a1.tick_params(labelsize=5.5)
    hcols = [c for c in O.columns if c.startswith("u ")]
    xs = [1.25, 1.75, 2.5]
    for m in ("within_shape", "llm_B", "haiku_B", "ensemble", "lib_shape", "chronos2", "own_linear"):
        lab, c, ls = M[m]
        a2.plot(xs, O.loc[m, hcols], color=c, ls=ls, marker="o", ms=2.5, lw=1, label=lab)
    a2.set_xticks(xs, ["1-1.5", "1.5-2", "2-3"])
    a2.set_xlabel("horizon, multiples of t90", fontsize=6)
    a2.set_ylabel("error (SOH points)", fontsize=6)
    a2.set_title("Error grows with horizon", fontsize=6.5)
    a2.legend(fontsize=5, frameon=False)
    a2.tick_params(labelsize=5.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    F = pd.read_csv(z.LIB / "zero_shot_forecasts.csv")
    _, curves = z.load_library()
    fig_examples(F, curves, OUT / "figs" / "zero_shot_llm_examples.png")
    fig_errors(OUT / "figs" / "zero_shot_llm_errors.png")


if __name__ == "__main__":
    main()
