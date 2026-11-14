"""Check the detected knees of a dataset whose knee is measured on slow check-ups (e.g. data/lsd1).

    python scripts/check_knees.py --data data/lsd1

For every cell (excluded cells included, so the exclusion can be reviewed):
* the knee used by the study (``knee_point``: from ``spec.knee_from``, e.g. the C/5 check-ups);
* for comparison, the knee of the per-cycle ageing capacity (Severson-style);
* whether the study knee is *clear*: fade after the breakpoint at least ``MIN_RATIO`` times faster
  than before, and the breakpoint between ``MIN_FRAC`` and ``MAX_FRAC`` of the end-of-life cycle.
  Cells without a clear knee are listed as candidates for ``exclude_cells`` in dataset.json.

Writes reports/<name>/knee_check.csv and figs/knee_check.png. Uses capacity curves only, never a
prediction.
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
from knee_analogs.core import (DatasetSpec, eol_cycle, knee_point, load_cells, reached_eol,  # noqa: E402
                               target_frame)

MIN_RATIO, MIN_FRAC, MAX_FRAC = 1.3, 0.2, 0.95
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4


def ageing_view(d: pd.DataFrame) -> pd.DataFrame:
    """The same cell with the knee measured on its per-cycle capacity instead."""
    e = d.copy()
    e.attrs = dict(d.attrs, spec=dataclasses.replace(d.attrs["spec"], knee_from="QD"))
    e.attrs.pop("ref", None)
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "lsd1"))
    args = ap.parse_args()
    data = Path(args.data)
    spec = DatasetSpec.from_folder(str(data))
    out = ROOT / "reports" / data.resolve().name
    (out / "figs").mkdir(parents=True, exist_ok=True)
    cells = load_cells(str(data), spec=dataclasses.replace(spec, exclude_cells={}, no_knee_cells={}))
    rows = []
    for k, d in cells.items():
        kn, s1, s2 = knee_point(d)
        eol = eol_cycle(d)
        ka = knee_point(ageing_view(d))[0]
        ratio = s2 / s1 if s1 else np.nan
        # the knee must sit inside the span it was fitted on: up to end of life, or the whole test
        span = eol if spec.knee_until_eol else float(target_frame(d)["cycle"].iloc[-1])
        clear = bool(ratio >= MIN_RATIO and MIN_FRAC <= kn / span <= MAX_FRAC)
        reason = "" if clear else ("no breakpoint with accelerating fade" if not np.isfinite(kn) else
                                   f"fade after/before breakpoint {ratio:.2f} < {MIN_RATIO}" if not ratio >= MIN_RATIO
                                   else f"breakpoint at {kn / span:.2f} of the fitted span")
        rows.append({"cell": k, "condition": d.attrs.get("condition", ""), "knee": kn, "knee_ageing_cycles": ka,
                     "eol": eol, "reached_eol": reached_eol(d), "knee_over_span": kn / span,
                     "slope_before_mAh": s1, "slope_after_mAh": s2, "slope_ratio": ratio, "clear_knee": clear,
                     "reason": reason, "excluded": k in spec.exclude_cells or k in spec.no_knee_cells,
                     "last_cycle": d["cycle"].iloc[-1]})
    R = pd.DataFrame(rows).set_index("cell")
    R.to_csv(out / "knee_check.csv")
    pd.set_option("display.width", 220)
    print(R.drop(columns=["reason"]).round(2).to_string())
    print(f"\n{(~R['clear_knee']).sum()} cells without a clear knee:")
    print({k: f"no clear knee ({r})" for k, r in R.loc[~R["clear_knee"], "reason"].items()})

    order = R.sort_values(["knee"]).index
    ncol = 7
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(170 * mm, 24 * mm * nrow), layout="constrained", squeeze=False)
    for ax, k in zip(axes.ravel(), order):
        d, r = cells[k], R.loc[k]
        t = target_frame(d)
        q0, qt0 = d.attrs["q_initial"], t.attrs["q_initial"]
        ax.plot(d["cycle"], d["QDs"] / q0, color="#c8c8c8", lw=0.6)
        ax.plot(t["cycle"], t["QDs"] / qt0, color="#009E73", lw=0.7, marker="o", ms=0.7)
        if np.isfinite(r["knee"]):
            ax.axvline(r["knee"], color="#D55E00" if r["clear_knee"] else "#6b6b6b", lw=0.9)
        ax.axvline(r["eol"], color="#6b6b6b", lw=0.6, ls="--")
        ax.axhline(spec.eol_fraction, color="#6b6b6b", lw=0.4, ls=":")
        ax.set_title(f"{k} {str(r['condition'])[:18]} knee {r['knee']:.0f}" + ("" if r["clear_knee"] else " UNCLEAR"),
                     fontsize=5.5, pad=1, color="black" if r["clear_knee"] else "#D55E00")
        ax.tick_params(labelsize=4.5)
        ax.set_ylim(min(0.6, float((t["QDs"] / qt0).min()) - 0.02), 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle("Capacity / initial: knee series (green) with knee (orange; grey = unclear) and end of life "
                 "(dashed); per-cycle ageing capacity in light grey", fontsize=7, x=0.01, ha="left")
    fig.savefig(out / "figs" / "knee_check.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
