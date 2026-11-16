"""Scenario B+: accelerated tests to 90 % SOH plus a short test of the use cell itself, stopped at 98 %
(primary) or 99 % SOH (method fixed in knee_analogs/scenario_b.py, forecast_early).

    python scripts/scenario_b_early.py

Writes reports/library/scenario_b_early.csv (MAE per type and level), scenario_b_early_ci.csv,
scenario_b_early_t90.csv and figs/scenario_b_early_examples.png. Needs the Scenario B LLM runs
(llm_trial/scenario_b_llm.py, original + 3 repeats) for the LLM rows; works without them.
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
from scenario_b_study import NAMES, stratified_ci  # noqa: E402

LLM = ROOT / "data" / "library" / "llm" / "scenario_b"
OUT = ROOT / "reports" / "library"
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
METHODS = ["early_accel", "early_accel+llm", "early_library", "early_linear", "speed_regression", "opus_data_med",
           "oracle_speed", "use_database"]
STYLE = {"early_accel": ("#D55E00", "-"), "early_accel+llm": ("#882255", "-"), "early_library": ("#E69F00", "--"),
         "early_linear": ("#0072B2", "--"), "speed_regression": ("#56B4E9", ":"), "opus_data_med": ("#CC79A7", ":"),
         "use_database": ("#009E73", "-")}


def opus_median():
    runs = [LLM / "opus_data.csv"] + sorted((LLM / "repeats").glob("opus_data_r*.csv"))
    runs = [pd.read_csv(f) for f in runs if f.exists()]
    if not runs:
        return None, None
    A = pd.concat(runs).groupby(["type", "key", "soh"], as_index=False)[["q10", "q50", "q90"]].median()
    t90 = {}
    for (typ, key), L in A.groupby(["type", "key"]):
        L = L.sort_values("soh")
        t90[(typ, key)] = float(np.interp(0.9, L["soh"], L["q50"]))
    return A, t90


def fig(P, curves, path):
    fig, axes = plt.subplots(1, len(NAMES), figsize=(180 * mm, 50 * mm), layout="constrained")
    for ax, typ in zip(axes, NAMES):
        D = P[P["type"] == typ]
        if D.empty:
            continue
        e = (D["early_accel"] - D["actual"]).abs().groupby(D["id"]).mean().sort_values()
        q = e.index[len(e) // 2]
        g = D[D["id"] == q].sort_values("efc")
        t, s = curves[q]
        ts = g["t_stop"].iloc[0]
        ax.axvspan(0, ts, color="k", alpha=0.06, lw=0)
        ax.plot(t[t <= ts], s[t <= ts] * 100, "k.", ms=2, label="short use test (seen)")
        ax.plot(t[t > ts], s[t > ts] * 100, ".", color="#a0a0a0", ms=1.5, label="actual (not seen)")
        for m in ("early_accel", "early_accel+llm", "early_linear", "use_database"):
            if m in g and g[m].notna().any():
                c, ls = STYLE[m]
                ax.plot(g["efc"], g[m] * 100, color=c, ls=ls, lw=0.9, label=m)
        ax.set_xlim(0, g["efc"].max() * 1.05)
        ax.set_ylim(max(min(s[t <= g["efc"].max()].min() * 100, g["early_linear"].min() * 100) - 2, 50), 101)
        ax.set_title(NAMES[typ], fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("EFC", fontsize=6)
    axes[0].set_ylabel("SOH (%)", fontsize=6)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=5, frameon=False)
    fig.suptitle("Accelerated tests to 90 % + a short test at the use condition to 98 % (shaded); "
                 "median-error cell of early_accel", fontsize=6.5, x=0.01, ha="left")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    meta, curves = z.load_library()
    A, llm_t90 = opus_median()
    res, cis, t90s = [], [], []
    pd.set_option("display.width", 220)
    for level in (0.98, 0.99):
        P = b.forecast_early(meta, curves, level, llm_t90)
        if A is not None:  # the Opus (data, median of runs) curve on the same points, for comparison
            P["opus_data_med"] = np.nan
            for (typ, key), idx in P.groupby(["type", "key"]).groups.items():
                L = A[(A["type"] == typ) & (A["key"] == key)].sort_values("soh", ascending=False)
                if len(L):
                    P.loc[idx, "opus_data_med"] = b.curve_from_levels(L["soh"], L["q50"], P.loc[idx, "efc"].to_numpy())
        T, O = b.score(P, METHODS)
        print(f"\n=== short use test to {level:.0%} SOH; MAE (SOH points) per type, common points\n"
              + T.round(2).to_string() + "\nmean over types:\n" + O.round(2).to_string())
        res.append(T.assign(level=level).reset_index())
        C = P.drop_duplicates("id")
        for k in [c for c in C if c.startswith("t90_") and c != "t90_true"]:
            f = 10 ** np.abs(np.log10(C[k] / C["t90_true"]))
            t90s.append({"level": level, "method": k[4:], **f.groupby(C["type"]).median().round(2).to_dict()})
        print("stop at (median fraction of true t90):", (C["t_stop"] / C["t90_true"]).groupby(C["type"]).median()
              .round(2).to_dict())
        for a, c in [("early_accel", "speed_regression"), ("early_accel", "early_library"),
                     ("early_accel", "early_linear"), ("early_accel+llm", "early_accel"),
                     ("early_accel", "use_database"), ("early_accel", "opus_data_med")]:
            if a in P and c in P:
                r = stratified_ci(P, a, c)
                cis.append({"level": level, "a": a, "b": c, **r})
                print(f"  {a} - {c}: {r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")
        if level == 0.98:
            fig(P, curves, OUT / "figs" / "scenario_b_early_examples.png")
    pd.concat(res).to_csv(OUT / "scenario_b_early.csv", index=False)
    pd.DataFrame(cis).to_csv(OUT / "scenario_b_early_ci.csv", index=False)
    S = pd.DataFrame(t90s)
    S.to_csv(OUT / "scenario_b_early_t90.csv", index=False)
    print("\nmedian factor between predicted and true t90\n" + S.to_string(index=False))


if __name__ == "__main__":
    main()
