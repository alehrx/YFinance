"""Battery knee prediction with ageing analogs, leave-one-cell-out.

    python run_study.py --data data/severson_b2

The folder holds one CSV per cell with columns cycle, QD (discharge capacity, Ah) and optionally
IR, Tavg, chargetime, charge_policy. Default data: Severson et al. 2019, batch 2017-06-30
(get it with `python scripts/fetch_severson_b2.py`).

Writes reports/<dataset folder name>/*.csv and reports/<dataset folder name>/figs/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from knee_analogs import evaluate as ev
from knee_analogs.core import eol_cycle, knee_point, load_cells

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "severson_b2"   # set from --data in main()
FIG = OUT / "figs"
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
BLUE, VERM, GREEN, ORANGE, PINK, SKY, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#6b6b6b"
LABEL = {"mean": "average of other cells", "protocol_regression": "regression on protocol",
         "early_regression": "regression on early-cycle features", "analogs_features": "analogs: early-cycle features",
         "analogs_curve": "analogs: curve shape", "analogs_combined": "analogs: features + curve"}
COLOR = {"mean": GREY, "protocol_regression": SKY, "early_regression": GREEN, "analogs_features": VERM,
         "analogs_curve": ORANGE, "analogs_combined": PINK}


def short(c):
    return c.replace("2017_06_30_", "")


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)


def fig_knees(cells):
    rows = sorted(((knee_point(d)[0], k) for k, d in cells.items()))
    fig, axes = plt.subplots(6, 7, figsize=(170 * mm, 150 * mm), layout="constrained", sharey=True)
    for ax, (kn, k) in zip(axes.ravel(), rows):
        d = cells[k]
        ax.plot(d["cycle"], d["QD"], color="#c8c8c8", lw=0.5)
        ax.plot(d["cycle"], d["QDs"], color=BLUE, lw=0.9)
        ax.axvline(kn, color=VERM, lw=0.9)
        ax.axvline(eol_cycle(d), color=GREY, lw=0.6, ls="--")
        ax.set_title(f"{short(k)}  knee {kn:.0f}", fontsize=6, pad=1)
        ax.tick_params(labelsize=5); ax.set_ylim(0.8, 1.1); clean(ax)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("Detected knee (orange) and 80% end of life (dashed), 42 cells; capacity in Ah vs cycle",
                 fontsize=8, x=0.01, ha="left")
    save(fig, "knees_detected")


def fig_testA(curves_by_n, P100):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(170 * mm, 72 * mm), layout="constrained")
    ns = sorted(curves_by_n)
    for m in ["mean", "protocol_regression", "early_regression", "analogs_curve", "analogs_features"]:
        a1.plot(ns, [curves_by_n[n].loc[m, "MAE_cycles"] for n in ns], "o-", color=COLOR[m], label=LABEL[m], ms=4)
    a1.set_xlabel("cycles observed before predicting")
    a1.set_ylabel("mean abs. error of knee cycle")
    a1.set_xticks(ns); a1.tick_params(axis="x", which="minor", bottom=False)
    a1.set_ylim(0, None)
    a1.legend(fontsize=7, loc="lower left")
    clean(a1)
    lim = [50, 450]
    a2.plot(lim, lim, color=GREY, lw=0.8, ls="--")
    for m in ["early_regression", "analogs_features"]:
        mae = (P100[m] - P100["actual"]).abs().mean()
        a2.scatter(P100["actual"], P100[m], s=16, color=COLOR[m], alpha=0.85,
                   label=f"{LABEL[m]} (MAE {mae:.0f})")
    a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel("actual knee cycle"); a2.set_ylabel("predicted from first 100 cycles")
    a2.legend(fontsize=7, loc="lower right")
    clean(a2)
    save(fig, "testA_knee_prediction")


def fig_examples(cells, P100, picks):
    fig, axes = plt.subplots(1, len(picks), figsize=(170 * mm, 68 * mm), layout="constrained", sharey=True)
    for ax, c in zip(axes, picks):
        r = P100.loc[c]
        for j, nb in enumerate(r["analogs_features_neighbors"].split(",")):
            d = cells[nb]
            ax.plot(d["cycle"], d["QDs"], color=SKY, lw=0.8, alpha=0.9, label="matched cells (full life)" if j == 0 else None)
        d = cells[c]
        ax.plot(d["cycle"], d["QDs"], color=VERM, lw=1.0, ls=(0, (3, 2)), label="held-out cell, rest of life")
        e = d[d["cycle"] <= 100]
        ax.plot(e["cycle"], e["QDs"], color="black", lw=1.8, label="what the method saw (first 100)")
        ax.axvspan(r["analogs_features_lo"], r["analogs_features_hi"], color=SKY, alpha=0.18, lw=0,
                   label="knee range of matches")
        ax.axvline(r["analogs_features"], color=BLUE, lw=1.0, label="predicted knee")
        ax.axvline(r["actual"], color=VERM, lw=1.0, label="actual knee")
        ax.set_title(f"{short(c)}: predicted {r['analogs_features']:.0f}, actual {r['actual']:.0f}", fontsize=8.5, loc="left")
        ax.set_xlabel("cycle"); ax.set_ylim(0.8, 1.1); clean(ax)
    axes[0].set_ylabel("discharge capacity (Ah)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    save(fig, "testA_examples")


def fig_testB(W, summ):
    fig, ax = plt.subplots(figsize=(120 * mm, 70 * mm), layout="constrained")
    g = W.groupby("cycle")
    for m, col, lab in [("cycle_count_only", GREY, "cycle count only"), ("regression", GREEN, "regression on state"),
                        ("analogs", VERM, "analogs: similar state in other cells")]:
        ax.plot(g.size().index, g.apply(lambda x: (x[m] - x["actual"]).abs().mean()), "o-", color=col, ms=4,
                label=f"{lab}")
    ax.set_xlabel("checkpoint cycle"); ax.set_ylabel("mean abs. error of\ncycles left until the knee")
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title(f"AUC for 'knee within 50 cycles': analogs {summ['AUC_soon_analogs']:.2f}, "
                 f"regression {summ['AUC_soon_regression']:.2f}, cycle count {summ['AUC_soon_cycle_count_only']:.2f}",
                 fontsize=8, loc="left")
    clean(ax)
    save(fig, "testB_early_warning")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "severson_b2"))
    ap.add_argument("--name", help="report folder name (default: the data folder's name)")
    args = ap.parse_args()
    global OUT, FIG
    OUT = ROOT / "reports" / (args.name or Path(args.data).resolve().name)
    FIG = OUT / "figs"
    FIG.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    cells = load_cells(args.data)
    print(f"{len(cells)} cells")
    fig_knees(cells)

    summaries, P100 = {}, None
    for n in (50, 100, 150, 200):
        T, C = ev.cell_table(cells, n)
        P = ev.loco_knee(T, C)
        summaries[n] = ev.summarize(P)
        P.to_csv(OUT / f"testA_predictions_first{n}.csv")
        if n == 100:
            P100 = P
        print(f"\nknee from first {n} cycles\n" + summaries[n].round(1).to_string())
    pd.concat(summaries, names=["first_n_cycles"]).to_csv(OUT / "testA_summary.csv")
    T, C = ev.cell_table(cells, 100)
    pe = ev.summarize(ev.loco_knee(T, C, target="eol"))
    pe.to_csv(OUT / "testA_eol_first100_summary.csv")
    print("\n80% end of life from first 100 cycles\n" + pe.round(1).to_string())
    fig_testA(summaries, P100)
    err = (P100["analogs_features"] - P100["actual"]).abs().sort_values()
    picks = [err.index[len(err) // 2], P100["actual"].idxmin(), err.index[-1]]  # typical, short-lived, worst
    fig_examples(cells, P100, picks)

    S = ev.snapshots(cells)
    W = ev.loco_warning(S)
    W.to_csv(OUT / "testB_early_warning.csv", index=False)
    summ = ev.warning_summary(W)
    pd.Series(summ).to_csv(OUT / "testB_summary.csv")
    print("\nearly warning:", {k: round(float(v), 3) for k, v in summ.items()})
    fig_testB(W, summ)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
