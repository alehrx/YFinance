"""Battery knee prediction with ageing analogs, leave-one-cell-out.

    python run_study.py --data data/severson_b2

The folder holds one CSV per cell with columns cycle, QD (discharge capacity, Ah) and optionally
IR, Tavg, chargetime, charge_policy, condition, plus an optional dataset.json with the
per-dataset settings (see knee_analogs.core.DatasetSpec). Default data: Severson et al. 2019,
batch 2017-06-30 (get it with `python scripts/fetch_severson_b2.py`).

Writes reports/<dataset folder name>/*.csv and reports/<dataset folder name>/figs/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from knee_analogs import evaluate as ev
from knee_analogs.core import eol_cycle, knee_point, load_cells, near_duplicates, reached_eol, spec_of

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


def ylim(cells):
    spec = spec_of(next(iter(cells.values())))
    top = max(spec.nominal_ah, max(d["QDs"].iloc[:10].median() for d in cells.values()))
    return spec.eol_ah - 0.07 * spec.nominal_ah, top * 1.005


def fig_knees(cells):
    rows = sorted(((knee_point(d)[0], k) for k, d in cells.items()), key=lambda r: (np.nan_to_num(r[0], nan=1e9), r[1]))
    ncol = 7
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(170 * mm, 25 * mm * nrow), layout="constrained", sharey=True,
                             squeeze=False)
    yl = ylim(cells)
    for ax, (kn, k) in zip(axes.ravel(), rows):
        d = cells[k]
        ax.plot(d["cycle"], d["QD"], color="#c8c8c8", lw=0.5)
        ax.plot(d["cycle"], d["QDs"], color=BLUE, lw=0.9, marker="o" if spec_of(d).sparse else None, ms=1.5)
        ax.axvline(kn, color=VERM, lw=0.9)
        if reached_eol(d):
            ax.axvline(eol_cycle(d), color=GREY, lw=0.6, ls="--")
        ax.set_title(f"{short(k)}  knee {kn:.0f}" + ("" if reached_eol(d) else " (no EOL)"), fontsize=6, pad=1)
        ax.tick_params(labelsize=5); ax.set_ylim(*yl); clean(ax)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    eolp = spec_of(next(iter(cells.values()))).eol_fraction * 100
    fig.suptitle(f"Detected knee (orange) and {eolp:.0f}% end of life (dashed), {len(rows)} cells; capacity in Ah vs cycle",
                 fontsize=8, x=0.01, ha="left")
    save(fig, "knees_detected")


def fig_testA(curves_by_n, P100, main_n):
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
    v = P100[["actual", "early_regression", "analogs_features"]].to_numpy()
    lim = [0.9 * np.nanmin(v), 1.05 * np.nanmax(v)]
    a2.plot(lim, lim, color=GREY, lw=0.8, ls="--")
    for m in ["early_regression", "analogs_features"]:
        mae = (P100[m] - P100["actual"]).abs().mean()
        a2.scatter(P100["actual"], P100[m], s=16, color=COLOR[m], alpha=0.85,
                   label=f"{LABEL[m]} (MAE {mae:.0f})")
    a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel("actual knee cycle"); a2.set_ylabel(f"predicted from first {main_n} cycles")
    a2.legend(fontsize=7, loc="lower right")
    clean(a2)
    save(fig, "testA_knee_prediction")


def fig_examples(cells, P100, picks, main_n):
    fig, axes = plt.subplots(1, len(picks), figsize=(170 * mm, 68 * mm), layout="constrained", sharey=True)
    for ax, c in zip(axes, picks):
        r = P100.loc[c]
        for j, nb in enumerate(r["analogs_features_neighbors"].split(",")):
            d = cells[nb]
            ax.plot(d["cycle"], d["QDs"], color=SKY, lw=0.8, alpha=0.9, label="matched cells (full life)" if j == 0 else None)
        d = cells[c]
        ax.plot(d["cycle"], d["QDs"], color=VERM, lw=1.0, ls=(0, (3, 2)), label="held-out cell, rest of life")
        e = d[d["cycle"] <= main_n]
        ax.plot(e["cycle"], e["QDs"], color="black", lw=1.8, label=f"what the method saw (first {main_n})")
        ax.axvspan(r["analogs_features_lo"], r["analogs_features_hi"], color=SKY, alpha=0.18, lw=0,
                   label="knee range of matches")
        ax.axvline(r["analogs_features"], color=BLUE, lw=1.0, label="predicted knee")
        ax.axvline(r["actual"], color=VERM, lw=1.0, label="actual knee")
        ax.set_title(f"{short(c)}: predicted {r['analogs_features']:.0f}, actual {r['actual']:.0f}", fontsize=8.5, loc="left")
        ax.set_xlabel("cycle"); ax.set_ylim(*ylim(cells)); clean(ax)
    axes[0].set_ylabel("discharge capacity (Ah)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    save(fig, "testA_examples")


def fig_testB(W, summ, window):
    fig, ax = plt.subplots(figsize=(120 * mm, 70 * mm), layout="constrained")
    g = W.groupby("cycle")
    for m, col, lab in [("cycle_count_only", GREY, "cycle count only"), ("regression", GREEN, "regression on state"),
                        ("analogs", VERM, "analogs: similar state in other cells")]:
        ax.plot(g.size().index, g.apply(lambda x: (x[m] - x["actual"]).abs().mean()), "o-", color=col, ms=4,
                label=f"{lab}")
    ax.set_xlabel("checkpoint cycle"); ax.set_ylabel("mean abs. error of\ncycles left until the knee")
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title(f"AUC for 'knee within {window} cycles': analogs {summ['AUC_soon_analogs']:.2f}, "
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
    spec = spec_of(next(iter(cells.values())))
    print(f"{len(cells)} cells; nominal {spec.nominal_ah} Ah, end of life at {spec.eol_fraction:.0%}"
          f"{', sparse check-ups' if spec.sparse else ''}")
    dups = near_duplicates(cells)
    if dups:
        print("WARNING near-identical cells (check before trusting any result):", dups)
    pd.DataFrame(dups, columns=["cell_a", "cell_b", "rms_Ah"]).to_csv(OUT / "near_duplicates.csv", index=False)
    knees = pd.DataFrame([{"cell": k, "knee": knee_point(d)[0], "eol": eol_cycle(d), "reached_eol": reached_eol(d),
                           "condition": d.attrs.get("condition", ""), "last_cycle": d["cycle"].iloc[-1],
                           "points": len(d)} for k, d in cells.items()]).set_index("cell")
    knees.to_csv(OUT / "knees.csv")
    print(f"{int(knees['reached_eol'].sum())} of {len(knees)} cells reached end of life")
    fig_knees(cells)

    summaries, details, P100 = {}, {}, None
    for n in spec.early_cycles:
        T, C = ev.cell_table(cells, n)
        P = ev.loco_knee(T, C)
        summaries[n] = ev.summarize(P)
        details[n] = ev.summarize_detail(P)
        P.to_csv(OUT / f"testA_predictions_first{n}.csv")
        if n == spec.main_n:
            P100 = P
        bd = details[n].attrs.get("analogs_minus_regression", {})
        print(f"\nknee from first {n} cycles\n" + details[n].round(2).to_string()
              + (f"\nanalogs - regression MAE: {bd['diff']:.1f} cycles, 95% CI [{bd['ci_lo']:.1f}, {bd['ci_hi']:.1f}]"
                 if bd else ""))
    pd.concat(summaries, names=["first_n_cycles"]).to_csv(OUT / "testA_summary.csv")
    pd.concat(details, names=["first_n_cycles"]).to_csv(OUT / "testA_detail.csv")
    pd.DataFrame({n: details[n].attrs.get("analogs_minus_regression", {}) for n in details}).T.rename_axis(
        "first_n_cycles").to_csv(OUT / "testA_analogs_minus_regression_ci.csv")
    T, C = ev.cell_table(cells, spec.main_n)
    pe = ev.summarize(ev.loco_knee(T, C, target="eol"))
    pe.to_csv(OUT / f"testA_eol_first{spec.main_n}_summary.csv")
    print(f"\nend of life from first {spec.main_n} cycles\n" + pe.round(1).to_string())
    n_cond = T["condition"].value_counts()
    if (n_cond >= 2).sum() >= 2:
        Pc = ev.loco_knee(T, C, leave_out="condition")
        Pc.to_csv(OUT / f"testA_predictions_first{spec.main_n}_leave_condition_out.csv")
        dc = ev.summarize_detail(Pc)
        dc.to_csv(OUT / f"testA_detail_first{spec.main_n}_leave_condition_out.csv")
        print(f"\nleave one condition out ({len(n_cond)} conditions), first {spec.main_n} cycles\n" + dc.round(2).to_string())
    fig_testA(summaries, P100, spec.main_n)
    err = (P100["analogs_features"] - P100["actual"]).abs().sort_values()
    picks = [err.index[len(err) // 2], P100["actual"].idxmin(), err.index[-1]]  # typical, short-lived, worst
    fig_examples(cells, P100, picks, spec.main_n)

    S = ev.snapshots(cells)
    W = ev.loco_warning(S, checkpoints=spec.checkpoints, window=spec.warning_window, sparse=spec.sparse)
    W.to_csv(OUT / "testB_early_warning.csv", index=False)
    summ = ev.warning_summary(W, window=spec.warning_window)
    pd.Series(summ).to_csv(OUT / "testB_summary.csv")
    print("\nearly warning:", {k: round(float(v), 3) for k, v in summ.items()})
    if (S.groupby("condition")["cell"].nunique() >= 2).sum() >= 2:
        Wc = ev.loco_warning(S, checkpoints=spec.checkpoints, window=spec.warning_window, sparse=spec.sparse,
                             leave_out="condition")
        Wc.to_csv(OUT / "testB_early_warning_leave_condition_out.csv", index=False)
        print("early warning, leave one condition out:",
              {k: round(float(v), 3) for k, v in ev.warning_summary(Wc, window=spec.warning_window).items()})
    fig_testB(W, summ, spec.warning_window)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
