"""Figures of Scenario B+ forecasts (knee_analogs/scenario_b.py, forecast_early): the use cell is tested at its
use condition until SOH reaches the stop level; the rest of its curve (to 3 x t90) is forecast.

    python scripts/fig_scenario_b_predictions.py            # --replot: reuse the cached forecasts

1. scenario_b_predictions_by_stop.png: one cell per type (the median-error cell of slope_accel at the 96 %
   stop), with the use test stopped at 98, 97, 96 and 95 % SOH (rows).
2. scenario_b_predictions_spread.png: stop at 96 %; per type the cells at the 10th, 50th and 90th
   percentile of the slope_accel error (good, typical, bad).

The per-cell forecasts are cached in data/library/scenario_b_early/ (gitignored).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from knee_analogs import scenario_b as b  # noqa: E402
from knee_analogs import zero_shot as z  # noqa: E402
from scenario_b_study import NAMES  # noqa: E402

OUT = ROOT / "reports" / "library" / "figs"
CACHE = ROOT / "data" / "library" / "scenario_b_early"
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
LEVELS = (0.98, 0.97, 0.96, 0.95)
REF_LEVEL = 0.96
STYLE = {  # colour, line style, label
    "slope_accel": ("#D55E00", "-", "slope scaling, accelerated cells' ratio"),
    "slope_library": ("#E69F00", "--", "slope scaling, library ratio"),
    "early_linear": ("#0072B2", ":", "own straight line"),
    "use_database": ("#009E73", "-.", "use-condition database (other cells)"),
}
FS, SMALL = 6, 5


def forecasts(replot):
    CACHE.mkdir(parents=True, exist_ok=True)
    P = {}
    for level in LEVELS:
        f = CACHE / f"forecast_{level:.2f}.csv.gz"
        if replot and f.exists():
            P[level] = pd.read_csv(f)
        else:
            P[level] = b.forecast_early(*z.load_library(), level)
            P[level].to_csv(f, index=False)
        print(f"{level:.0%}: {P[level]['id'].nunique()} cells", flush=True)
    return P


def cell_error(D, m="slope_accel"):
    return (D[m] - D["actual"]).abs().groupby(D["id"]).mean() * 100


def panel(ax, curves, D, cid, show_err=True):
    g = D[D["id"] == cid].sort_values("efc")
    t, s = curves[cid]
    ts, tend = g["t_stop"].iloc[0], g["efc"].max()
    ax.axvspan(0, ts, color="k", alpha=0.07, lw=0)
    ax.plot(t[t <= ts], s[t <= ts] * 100, "k.", ms=1.6, label="use test (seen)")
    ax.plot(t[(t > ts) & (t <= tend)], s[(t > ts) & (t <= tend)] * 100, ".", color="#9a9a9a", ms=1.3,
            label="actual (not seen)")
    for m, (c, ls, lab) in STYLE.items():
        if g[m].notna().any():
            ax.plot(g["efc"], g[m] * 100, color=c, ls=ls, lw=0.9, label=lab)
    lo = np.nanmin([s[t <= tend].min()] + [g[m].min() for m in STYLE]) * 100
    ax.set_ylim(max(lo - 2, 40), 101)
    ax.set_xlim(0, tend * 1.03)
    ax.tick_params(labelsize=SMALL, length=2)
    if show_err:
        e = {m: (g[m] - g["actual"]).abs().mean() * 100 for m in ("slope_accel", "use_database")}
        ax.text(0.97, 0.95, f"slope {e['slope_accel']:.1f}\ndatabase {e['use_database']:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=SMALL, color="0.25", linespacing=1.1)


def finish(fig, axes, path, title):
    for ax in axes[-1]:
        ax.set_xlabel("EFC", fontsize=FS)
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH (%)", fontsize=FS)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=SMALL, frameon=False)
    fig.suptitle(title, fontsize=FS + 0.5, x=0.01, ha="left")
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    _, curves = z.load_library()
    P = forecasts("--replot" in sys.argv)
    types = [t for t in NAMES if t in set(P[REF_LEVEL]["type"])]
    common = set.intersection(*[set(P[lv]["id"]) for lv in LEVELS])

    # 1. the same typical cell per type, the use test stopped later and later
    fig, axes = plt.subplots(len(LEVELS), len(types), figsize=(180 * mm, 165 * mm), layout="constrained",
                             squeeze=False)
    D = P[REF_LEVEL]
    for j, typ in enumerate(types):
        e = cell_error(D[(D["type"] == typ) & D["id"].isin(common)]).sort_values()
        cid = e.index[len(e) // 2]
        for i, lv in enumerate(LEVELS):
            panel(axes[i, j], curves, P[lv], cid)
            if i == 0:
                axes[i, j].set_title(NAMES[typ], fontsize=FS)
            if j == 0:
                axes[i, j].text(-0.42, 0.5, f"stop at {lv:.0%}", transform=axes[i, j].transAxes, rotation=90,
                                ha="center", va="center", fontsize=FS, weight="bold")
        xmax = max(ax.get_xlim()[1] for ax in axes[:, j])
        ymin = min(ax.get_ylim()[0] for ax in axes[:, j])
        for ax in axes[:, j]:
            ax.set_xlim(0, xmax)
            ax.set_ylim(ymin, 101)
    finish(fig, axes, OUT / "scenario_b_predictions_by_stop.png",
           "Forecasts after a short test at the use condition (shaded), stopped later and later; one typical cell "
           "per type; MAE in SOH points")

    # 2. good, typical and bad cells at the 96 % stop
    fig, axes = plt.subplots(3, len(types), figsize=(180 * mm, 125 * mm), layout="constrained", squeeze=False)
    for j, typ in enumerate(types):
        e = cell_error(D[D["type"] == typ]).sort_values()
        for i, (q, name) in enumerate([(0.1, "good (10th pct.)"), (0.5, "typical (median)"), (0.9, "bad (90th pct.)")]):
            cid = e.index[min(int(round(q * (len(e) - 1))), len(e) - 1)]
            panel(axes[i, j], curves, D, cid)
            if i == 0:
                axes[i, j].set_title(f"{NAMES[typ]} ({len(e)} cells)", fontsize=FS)
            if j == 0:
                axes[i, j].text(-0.42, 0.5, name, transform=axes[i, j].transAxes, rotation=90, ha="center",
                                va="center", fontsize=FS, weight="bold")
    finish(fig, axes, OUT / "scenario_b_predictions_spread.png",
           "Use test stopped at 96 % SOH (shaded): good, typical and bad cells by the error of slope scaling "
           "(accelerated cells' ratio); MAE in SOH points")


if __name__ == "__main__":
    main()
