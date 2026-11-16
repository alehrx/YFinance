"""Forecast every cell's whole future ageing trajectory from its first N cycles (leave-one-cell-out).

    python scripts/trajectory_study.py --data data/mohtat
    python scripts/trajectory_study.py --data data/lsd1

Method and scoring: knee_analogs/trajectory.py (fixed before the first run). The primary method is
``analogs_state``; it is compared with every baseline by a bootstrap CI over cells. A knee is not
needed, so cells excluded from the knee tests (``exclude_cells``, ``no_knee_cells``) are used here.
N = the dataset's ``early_cycles``.

Writes reports/<name>/trajectory_summary.csv, trajectory_ci.csv, figs/trajectory_*.png, and the
per-check-up forecasts trajectory_first<N>.csv (not committed: they contain the measured curves).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee_analogs import trajectory as tj  # noqa: E402
from knee_analogs.core import DatasetSpec, load_cells  # noqa: E402

plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
PRIMARY = "analogs_state"
STYLE = {"analogs_state": ("#D55E00", "-", "analogs, matched state"), "analogs_cycle": ("#E69F00", "-", "analogs, same cycle"),
         "own_linear": ("#0072B2", "--", "own linear trend"), "own_powerlaw": ("#56B4E9", "--", "own power-law trend"),
         "population": ("#6b6b6b", ":", "average of other cells"), "same_condition": ("#999933", ":", "average, same condition"),
         "regression": ("#009E73", "-.", "ridge regression")}
MIN_CELLS = 8


def fig_examples(cells, F, n, out, name):
    errs = (F[PRIMARY] - F["actual"]).abs().groupby(F["cell"]).mean().dropna().sort_values()
    if errs.empty:
        return
    picks = [errs.index[len(errs) // 4], errs.index[len(errs) // 2], errs.index[(3 * len(errs)) // 4], errs.index[-1]]
    fig, axes = plt.subplots(1, 5, figsize=(180 * mm, 50 * mm), layout="constrained",
                             gridspec_kw={"width_ratios": [1.3, 1, 1, 1, 1]})
    S = tj.score(F)
    hcols = [c for c in S.columns if c.startswith("MAE_horizon_")]
    for m in S.index:
        c, ls, lab = STYLE[m]
        axes[0].plot(range(len(hcols)), S.loc[m, hcols].to_numpy(float), color=c, ls=ls, marker="o", ms=2.5, label=lab)
    axes[0].set_xticks(range(len(hcols)), [h.replace("MAE_horizon_", "") for h in hcols], fontsize=5, rotation=30)
    axes[0].set_xlabel("cycles ahead"); axes[0].set_ylabel("SOH error (% points)"); axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(plt.NullFormatter())
    axes[0].legend(fontsize=4.8, frameon=False)
    for ax, c in zip(axes[1:], picks):
        x, y = tj.soh_series(cells[c])
        g = F[F["cell"] == c]
        ax.plot(x[x <= n], y[x <= n] * 100, "k.", ms=2.5, label="seen (first N)")
        ax.plot(x[x > n], y[x > n] * 100, ".", color="#9a9a9a", ms=2, label="actual future")
        ax.fill_between(g["cycle"], g[f"{PRIMARY}_lo"] * 100, g[f"{PRIMARY}_hi"] * 100, color="#D55E00", alpha=0.15, lw=0)
        for m in (PRIMARY, "own_linear", "population"):
            col, ls, lab = STYLE[m]
            ax.plot(g["cycle"], g[m] * 100, color=col, ls=ls, lw=0.9, label=lab)
        ax.axvline(n, color="k", lw=0.5, ls=":")
        e = (g[PRIMARY] - g["actual"]).abs().mean() * 100
        ax.set_title(f"{c}\n{g['condition'].iloc[0]}\nerror {e:.1f} pts", fontsize=5)
        ax.tick_params(labelsize=5); ax.set_xlabel("cycle", fontsize=6)
    axes[1].set_ylabel("SOH (%)", fontsize=6)
    axes[-1].legend(fontsize=4.5, frameon=False, loc="lower left")
    fig.suptitle(f"{name}: whole-trajectory forecast from the first {n:g} cycles (band = range of the 5 analogs)",
                 fontsize=7, x=0.01, ha="left")
    fig.savefig(out / "figs" / f"trajectory_first{n:g}.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    spec = DatasetSpec.from_folder(str(data))
    out = ROOT / "reports" / data.resolve().name
    (out / "figs").mkdir(parents=True, exist_ok=True)
    cells = load_cells(str(data), spec=dataclasses.replace(spec, exclude_cells={}, no_knee_cells={}))
    print(f"{len(cells)} cells")
    summaries, cis = {}, []
    for n in spec.early_cycles:
        F = tj.forecast(cells, n)
        nq = F["cell"].nunique() if len(F) else 0
        if nq < MIN_CELLS:
            print(f"first {n}: only {nq} cells can be forecast (need {MIN_CELLS}); skipped")
            continue
        F.to_csv(out / f"trajectory_first{n:g}.csv", index=False)
        S = tj.score(F)
        summaries[n] = S
        print(f"\nfirst {n} cycles: {nq} cells\n" + S.round(2).to_string())
        for b in tj.METHODS:
            if b != PRIMARY and b in F and F[b].notna().any():
                r = tj.bootstrap_vs(F, PRIMARY, b)
                cis.append({"first_n": n, "vs": b, **r})
                print(f"  {PRIMARY} - {b}: {r['diff']:+.2f} pts [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] ({r['n_cells']} cells)")
        fig_examples(cells, F, n, out, data.resolve().name)
    pd.concat(summaries, names=["first_n"]).to_csv(out / "trajectory_summary.csv")
    pd.DataFrame(cis).to_csv(out / "trajectory_ci.csv", index=False)


if __name__ == "__main__":
    main()
