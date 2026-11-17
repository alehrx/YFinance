"""Figure: the trade-off between the length of the short test at the use condition and the forecast error
(Scenario B / B+). Descriptive: the stop levels 97 % and 96 % are added between the two pre-registered ones
(98 %, 95 %) to draw the curve; the methods are unchanged (knee_analogs/scenario_b.py, forecast_early).

    python scripts/fig_scenario_b_tradeoff.py            # --replot: redraw from the saved table

x = length of the short use test as a fraction of the cell's true t90 (per type the median over cells, then the
mean over types); x = 0 is Scenario B without a use test (Opus, median of 4 runs, speed + shape). Every level is
scored on its own points (after the stop), so panel (b) also shows each method minus the use-condition
database on the same points, with 95 % stratified bootstrap intervals.

Writes reports/library/figs/scenario_b_tradeoff.png and reports/library/scenario_b_tradeoff.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from knee_analogs import scenario_b as b  # noqa: E402
from knee_analogs import zero_shot as z  # noqa: E402
from scenario_b_study import stratified_ci  # noqa: E402

OUT = ROOT / "reports" / "library"
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
LEVELS = (0.98, 0.97, 0.96, 0.95)
METHODS = ["slope_accel", "slope_library", "early_linear", "early_accel", "oracle_speed", "use_database"]
STYLE = {  # colour, marker, label
    "slope_accel": ("#D55E00", "o", "slope scaling, accelerated cells' ratio"),
    "slope_library": ("#E69F00", "s", "slope scaling, library ratio"),
    "early_linear": ("#0072B2", "^", "own straight line"),
    "early_accel": ("#CC79A7", "v", "time scaling, accelerated cells' ratio"),
}


def compute():
    meta, curves = z.load_library()
    rows = []
    for level in LEVELS:
        P = b.forecast_early(meta, curves, level)
        _, O = b.score(P, METHODS)
        C = P.drop_duplicates("id")
        x = (C["t_stop"] / C["t90_true"]).groupby(C["type"]).median().mean()
        for m in METHODS:
            r = stratified_ci(P, m, "use_database") if m != "use_database" else {}
            rows.append({"level": level, "x": x, "method": m, "MAE": O[m], **r})
        print(f"{level:.0%}: stop at {x:.2f} t90", flush=True)
    # x = 0: Scenario B without a use test (whole curve to 3 t90, cells that reach 90 %)
    R = pd.read_csv(OUT / "scenario_b_repeats.csv").set_index("method")
    ci = pd.read_csv(OUT / "scenario_b_repeats_ci.csv")
    ci = ci[(ci["a"] == "opus_data_med_speed+shape") & (ci["b"] == "use_database")].iloc[0]
    ov = pd.read_csv(OUT / "scenario_b_overall.csv", index_col=0)["MAE"]
    rows += [{"level": 1.0, "x": 0.0, "method": "llm_speed+shape", "MAE": R.loc["opus_data_med_speed+shape", "MAE"],
              "diff": ci["diff"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"]},
             {"level": 1.0, "x": 0.0, "method": "use_database", "MAE": ov["use_database"]},
             {"level": 1.0, "x": 0.0, "method": "oracle_speed", "MAE": ov["oracle_speed"]}]
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "scenario_b_tradeoff.csv", index=False)
    print(T.round(2).to_string(index=False))
    return T


def plot(T):
    fs, small = 6.5, 5.5
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(180 * mm, 78 * mm), layout="constrained")
    g = lambda m: T[(T["method"] == m)].sort_values("x")  # noqa: E731
    test = lambda m: g(m)[g(m)["x"] > 0]  # noqa: E731
    xlab = "use test length (fraction of time to 90 % SOH)"
    handles = []
    # references: the database and the true speed, on the same points as each stop (x > 0); at x = 0 the
    # whole curve is scored, so those points stand apart and are not joined to the line
    for m, ls, mk, lab in [("use_database", "--", "D", "use-condition database (a full ageing campaign)"),
                           ("oracle_speed", ":", "x", "true speed (reference, not available in practice)")]:
        h, = a1.plot(test(m)["x"], test(m)["MAE"], color="0.4", ls=ls, lw=0.9, label=lab)
        a1.plot(0, g(m)["MAE"].iloc[0], mk, color="0.4", ms=3, mfc="none", mew=0.7)
        handles.append(h)
    for m, (c, mk, lab) in STYLE.items():
        h, = a1.plot(test(m)["x"], test(m)["MAE"], color=c, marker=mk, ms=3.5,
                     lw=1.1 if m != "early_accel" else 0.8, alpha=1 if m != "early_accel" else 0.55, label=lab)
        handles.append(h)
    L = g("llm_speed+shape")
    h, = a1.plot(L["x"], L["MAE"], "*", color="#882255", ms=8, label="no use test: LLM speed + accelerated shape")
    handles.append(h)
    a1.annotate(f"{L['MAE'].iloc[0]:.1f}", (0, L["MAE"].iloc[0]), xytext=(5, 2), textcoords="offset points",
                fontsize=small, color="#882255")
    for (m, level), xy in {("slope_accel", 0.98): (-15, -1), ("slope_library", 0.98): (-15, -3),
                           ("slope_accel", 0.95): (5, -9), ("slope_library", 0.95): (5, 3)}.items():
        r = test(m)[test(m)["level"] == level].iloc[0]
        a1.annotate(f"{r['MAE']:.1f}", (r["x"], r["MAE"]), xytext=xy, textcoords="offset points",
                    fontsize=small, color=STYLE[m][0])
    for m, lab in [("use_database", "database"), ("oracle_speed", "true speed")]:
        a1.annotate(lab, (0, g(m)["MAE"].iloc[0]), xytext=(5, -2), textcoords="offset points", fontsize=small,
                    color="0.4")
    a1.text(0.0, 1.15, "x = 0: whole curve scored", fontsize=small, color="0.4")
    for level in (0.98, 0.97, 0.96, 0.95):
        x = T.loc[T["level"] == level, "x"].iloc[0]
        for ax in (a1, a2):
            ax.axvline(x, color="0.88", lw=0.6, zorder=0)
        a1.text(x, 6.75, f"{level:.0%}", fontsize=small, ha="center", va="top", color="0.4")
    a1.text(0.0, 6.75, "SOH at stop:", fontsize=small, ha="left", va="top", color="0.4")
    a1.set_ylim(0, 7)
    a1.set_ylabel("forecast error (MAE, SOH points)", fontsize=fs)
    a1.set_title("(a) error after the use test, mean over 5 cell types", loc="left", fontsize=fs)

    a2.axhline(0, color="0.4", ls="--", lw=0.9)
    a2.text(0.005, -0.12, "as good as the database", fontsize=small, ha="left", va="top", color="0.4")
    for m, (c, mk, _) in {**STYLE, "llm_speed+shape": ("#882255", "*", "")}.items():
        s = g(m)
        a2.errorbar(s["x"] + {"slope_accel": -0.007, "early_linear": 0.007, "early_accel": 0.014}.get(m, 0),
                    s["diff"], yerr=[s["diff"] - s["ci_lo"], s["ci_hi"] - s["diff"]], color=c, marker=mk,
                    ms=3.5 if mk != "*" else 8, lw=1.1 if mk != "*" else 0, elinewidth=0.7, capsize=1.5,
                    alpha=1 if m != "early_accel" else 0.55)
    a2.set_ylabel("error minus database error (SOH points)", fontsize=fs)
    a2.set_title("(b) relative to the database, same points, 95 % CI", loc="left", fontsize=fs)
    for ax in (a1, a2):
        ax.set_xlim(-0.03, 0.52)
        ax.set_xlabel(xlab, fontsize=fs)
        ax.tick_params(labelsize=small)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=small, frameon=False)
    fig.savefig(OUT / "figs" / "scenario_b_tradeoff.png", dpi=250)
    plt.close(fig)


def main():
    f = OUT / "scenario_b_tradeoff.csv"
    plot(pd.read_csv(f) if "--replot" in sys.argv and f.exists() else compute())


if __name__ == "__main__":
    main()
