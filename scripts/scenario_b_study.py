"""Scenario B: accelerated cells observed to 90 % SOH -> the whole curve at use conditions
(method fixed in knee_analogs/scenario_b.py).

    python scripts/scenario_b_study.py

Writes reports/library/scenario_b_by_type.csv, scenario_b_overall.csv, scenario_b_t90.csv,
scenario_b_ci.csv and figs/scenario_b_examples.png. Per-point forecasts stay in data/library/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee_analogs import scenario_b as b  # noqa: E402
from knee_analogs import zero_shot as z  # noqa: E402

OUT = ROOT / "reports" / "library"
WORK = ROOT / "data" / "library"
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
BASE = ["accel_as_is", "speed_regression"]
REF = ["oracle_speed", "use_database"]
STYLE = {"accel_as_is": ("#6b6b6b", ":"), "speed_regression": ("#0072B2", "--"), "oracle_speed": ("#009E73", "-."),
         "use_database": ("#009E73", "-"), "opus_data": ("#882255", "-"), "opus_know": ("#CC6677", (0, (3, 1))),
         "haiku_data": ("#882255", (0, (1, 1))), "haiku_know": ("#CC6677", (0, (1, 1))),
         "opus_data_speed+shape": ("#E69F00", "-")}
NAMES = {"lsd1": "LSD", "mohtat": "Mohtat (NMC)", "tongji_nca": "Tongji NCA", "tongji_ncm": "Tongji NCM",
         "stanford_matrix": "Stanford matrix (NCA)"}


def stratified_ci(P, a, c, n_boot=5000, seed=0):
    """Mean over types of per-type MAE(a) - MAE(c), cells resampled within types."""
    G = P.dropna(subset=[a, c])
    per = [((D[a] - D["actual"]).abs().groupby(D["id"]).mean().to_numpy() * 100,
            (D[c] - D["actual"]).abs().groupby(D["id"]).mean().to_numpy() * 100) for _, D in G.groupby("type")]
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for ea, eb in per:
        idx = rng.integers(0, len(ea), (n_boot, len(ea)))
        boots += (ea[idx].mean(1) - eb[idx].mean(1)) / len(per)
    return {"diff": np.mean([ea.mean() - eb.mean() for ea, eb in per]), "ci_lo": np.percentile(boots, 2.5),
            "ci_hi": np.percentile(boots, 97.5), "types": len(per)}


def fig_examples(P, curves, methods, path):
    types = list(NAMES)
    fig, axes = plt.subplots(1, len(types), figsize=(180 * mm, 50 * mm), layout="constrained")
    for ax, typ in zip(axes, types):
        D = P[P["type"] == typ].dropna(subset=methods)
        if D.empty:
            continue
        e = (D[methods[0]] - D["actual"]).abs().groupby(D["id"]).mean().sort_values()
        q = e.index[len(e) // 2]
        g = D[D["id"] == q].sort_values("efc")
        t, s = curves[q]
        ax.plot(t, s * 100, ".", color="#a0a0a0", ms=1.5, label="actual (never seen)")
        for m in methods:
            c, ls = STYLE.get(m, ("k", "-"))
            ax.plot(np.r_[0, g["efc"]], np.r_[100, g[m] * 100], color=c, ls=ls, lw=0.9, label=m)
        ax.set_xlim(0, g["efc"].max() * 1.05)
        ax.set_title(NAMES[typ], fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("EFC", fontsize=6)
    axes[0].set_ylabel("SOH (%)", fontsize=6)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=5, frameon=False)
    fig.suptitle(f"A use-condition cell per type (median error of {methods[0]})", fontsize=7, x=0.01, ha="left")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    meta, curves = z.load_library()
    M = b.assign_types(meta)
    print(M.groupby(["type", "role"]).size().unstack().to_string(), flush=True)
    llm = {}
    for f in sorted((WORK / "llm" / "scenario_b").glob("*.csv")) if (WORK / "llm" / "scenario_b").exists() else []:
        llm[f.stem] = pd.read_csv(f)
    P, C = b.forecast(meta, curves, llm)
    P.to_csv(WORK / "scenario_b_forecasts.csv", index=False)
    post = [f"{m}_speed+shape" for m in llm]  # post hoc: LLM speed, shared shape
    methods = BASE + list(llm) + post + REF
    T, O = b.score(P, methods)
    T.to_csv(OUT / "scenario_b_by_type.csv")
    O.rename("MAE").to_csv(OUT / "scenario_b_overall.csv")
    pd.set_option("display.width", 220)
    print("\nMAE (SOH points) per type, common points:\n" + T.round(2).to_string())
    print("\nmean over types:\n" + O.round(2).to_string())
    # speed: how far off is the predicted t90, as a factor
    S = []
    for k in [c for c in C if c.startswith("t90_") and c != "t90_true"]:
        ok = C["t90_true"].notna() & C[k].notna()
        f = np.abs(np.log10(C.loc[ok, k] / C.loc[ok, "t90_true"]))
        S.append({"method": k[4:], **{NAMES.get(t, t): 10 ** f[C.loc[ok, "type"] == t].median() for t in NAMES},
                  "cells": int(ok.sum())})
    S = pd.DataFrame(S).set_index("method")
    S.to_csv(OUT / "scenario_b_t90.csv")
    print("\nmedian factor between predicted and true t90:\n" + S.round(2).to_string())
    acc = C.groupby("type")["n_acc"].first()
    ratio = {}
    for typ, D in M.groupby("type"):
        tt = {r: np.nanmedian([z.t_at(*curves[c]) for c in D.index[D["role"] == r]]) for r in ("accelerated", "use")}
        ratio[typ] = tt["use"] / tt["accelerated"]
    print("\nmedian t90 use / accelerated (true):", {k: round(v, 2) for k, v in ratio.items()}, "\naccelerated cells:",
          acc.to_dict())
    cis = []
    pairs = [("speed_regression", "accel_as_is")] + [(m, c) for m in llm for c in ("speed_regression", "accel_as_is")]
    pairs += [(m, "use_database") for m in BASE + list(llm)]
    if "opus_data" in llm and "opus_know" in llm:
        pairs.append(("opus_data", "opus_know"))
    if "haiku_data" in llm and "haiku_know" in llm:
        pairs.append(("haiku_data", "haiku_know"))
    if "haiku_data" in llm and "opus_data" in llm:
        pairs.append(("haiku_data", "opus_data"))
    pairs += [(f"{m}_speed+shape", c) for m in llm for c in (m, "speed_regression", "use_database")]
    for a, c in pairs:
        r = stratified_ci(P, a, c)
        cis.append({"a": a, "b": c, **r})
        print(f"  {a} - {c}: {r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] ({r['types']} types)")
    pd.DataFrame(cis).to_csv(OUT / "scenario_b_ci.csv", index=False)
    fig_examples(P, curves, ([m for m in ("opus_data",) if m in llm] + [m for m in ("opus_data_speed+shape",)
                                                                     if m in P] + BASE + ["use_database"]),
                 OUT / "figs" / "scenario_b_examples.png")


if __name__ == "__main__":
    main()
