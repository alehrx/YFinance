"""Do all cells age along the same curve, only at different speeds? Descriptive check on the
cross-dataset library (data/library, built by scripts/build_library.py).

    python scripts/universality.py

For every cell: t_x = first equivalent full cycle at which SOH falls to x (x = 95, 90, 85, 80, 75, 70 %;
interpolated, on the running minimum of SOH so a recovery does not count), and the ratios t_x / t_90.
If "same mechanisms, different speed" held exactly, every curve plotted against t / t_90 would be the
same, and t_80 / t_90 would be one number for all cells.

Writes reports/library/universality_cells.csv, universality_summary.csv and figs/universality.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data" / "library"
OUT = ROOT / "reports" / "library"
LEVELS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4


def crossing(t: np.ndarray, soh: np.ndarray, level: float) -> float:
    s = np.minimum.accumulate(soh)
    j = np.flatnonzero(s <= level)
    if not len(j):
        return np.nan
    j = j[0]
    if j == 0:
        return float(t[0])
    return float(t[j - 1] + (s[j - 1] - level) / (s[j - 1] - s[j]) * (t[j] - t[j - 1]))


def curve(cid):
    d = pd.read_csv(LIB / "curves" / f"{cid}.csv")
    return d["efc"].to_numpy(float), d["soh"].to_numpy(float)


def main():
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(LIB / "cells.csv")
    rows = []
    for _, m in meta.iterrows():
        t, s = curve(m["id"])
        r = {"id": m["id"], "dataset": m["dataset"], "chemistry": m["chemistry"], "T": m["T"], "chg_C": m["chg_C"],
             "dis_C": m["dis_C"], "condition": m["condition"]}
        for x in LEVELS:
            r[f"t{round(x * 100)}"] = crossing(t, s, x)
        rows.append(r)
    R = pd.DataFrame(rows)
    for x in (85, 80, 75, 70):
        R[f"r{x}"] = R[f"t{x}"] / R["t90"]
    R.to_csv(OUT / "universality_cells.csv", index=False)

    def iqr(v):
        v = v.dropna()
        return f"{v.median():.2f} [{v.quantile(.25):.2f}-{v.quantile(.75):.2f}] n={len(v)}" if len(v) else "-"

    S = R.groupby("dataset").agg(cells=("id", "size"), t90_median=("t90", "median"),
                                 t90_spread=("t90", lambda v: v.quantile(.9) / v.quantile(.1)),
                                 r85=("r85", iqr), r80=("r80", iqr), r75=("r75", iqr), r70=("r70", iqr))
    S.loc["all"] = [len(R), R["t90"].median(), R["t90"].quantile(.9) / R["t90"].quantile(.1),
                    iqr(R["r85"]), iqr(R["r80"]), iqr(R["r75"]), iqr(R["r70"])]
    S.to_csv(OUT / "universality_summary.csv")
    pd.set_option("display.width", 250)
    print(S.to_string())
    # how much of the spread of t80 does t90 explain? (log scale, across all cells reaching 80 %)
    Q = R.dropna(subset=["t90", "t80"])
    Q = Q[(Q["t90"] > 0)]
    c = np.corrcoef(np.log(Q["t90"]), np.log(Q["t80"]))[0, 1]
    print(f"\nlog t80 vs log t90 over {len(Q)} cells: r^2 = {c ** 2:.3f}; "
          f"t90 alone spans x{Q['t90'].quantile(.9) / Q['t90'].quantile(.1):.0f} (10-90th pct), "
          f"t80/t90 spans x{Q['r80'].quantile(.9) / Q['r80'].quantile(.1):.1f}")
    for ds, g in Q.groupby("dataset"):
        cc = np.corrcoef(np.log(g["t90"]), np.log(g["t80"]))[0, 1] if len(g) > 2 else np.nan
        print(f"  {ds:16s} n={len(g):3d}  r^2(log t80, log t90) = {cc ** 2:.2f}")

    dss = list(meta["dataset"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(180 * mm, 85 * mm), layout="constrained")
    grid = np.linspace(0, 3, 61)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, ds in enumerate(dss):
        ax = axes.ravel()[k]
        curves = []
        for _, m in meta[meta["dataset"] == ds].iterrows():
            t, s = curve(m["id"])
            t90 = R.loc[R["id"] == m["id"], "t90"].iloc[0]
            if not np.isfinite(t90) or t90 <= 0:
                continue
            ax.plot(t / t90, s * 100, color="#9a9a9a", lw=0.3, alpha=0.5)
            curves.append(np.interp(grid, t / t90, s, left=np.nan, right=np.nan))
        med = np.nanmedian(np.array(curves), 0)
        ax.plot(grid, med * 100, color=cols[k % len(cols)], lw=1.5)
        axes.ravel()[6].plot(grid, med * 100, color=cols[k % len(cols)], lw=1.2, label=ds)
        ax.set_title(f"{ds} ({len(curves)})", fontsize=7)
        ax.set_xlim(0, 3); ax.set_ylim(55, 102); ax.tick_params(labelsize=6)
        ax.axhline(90, color="k", lw=0.4, ls=":"); ax.axvline(1, color="k", lw=0.4, ls=":")
    a = axes.ravel()[6]
    a.set_title("median curves", fontsize=7); a.set_xlim(0, 3); a.set_ylim(55, 102); a.legend(fontsize=5, frameon=False)
    a.tick_params(labelsize=6)
    b = axes.ravel()[7]
    data = [R.loc[R["dataset"] == ds, "r80"].dropna() for ds in dss]
    b.boxplot(data, tick_labels=[d.replace("stanford_", "st_") for d in dss], showfliers=False)
    b.set_yscale("log"); b.set_title("t80 / t90 per dataset", fontsize=7); b.tick_params(labelsize=5, axis="x", rotation=40)
    b.tick_params(labelsize=6, axis="y")
    for ax in axes[1, :3]:
        ax.set_xlabel("EFC / EFC at 90 % SOH", fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH (%)", fontsize=6)
    fig.savefig(OUT / "figs" / "universality.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
