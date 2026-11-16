"""Scenario A: forecast each cell's future from its curve up to 90 % SOH, with no ageing data of its
own cell type (method fixed in knee_analogs/zero_shot.py).

    python scripts/zero_shot_study.py

Runs Chronos (scripts/chronos_forecast.py) in the Chronos environment, then every method, and writes
reports/library/zero_shot_by_dataset.csv, zero_shot_overall.csv, zero_shot_ci.csv and
figs/zero_shot_*.png. The per-point forecasts (data/library/zero_shot_forecasts.csv) stay local.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee_analogs import zero_shot as z  # noqa: E402

CHRONOS_PY = Path(r"C:\Users\alexa\anaconda3\envs\phm_env\python.exe")
MODELS = {"chronos2": "amazon/chronos-2", "chronos_bolt": "amazon/chronos-bolt-base"}
METHODS = ["lib_shape", "lib_shape_cond", "lib_median", "own_linear", "own_powerlaw", "chronos2", "chronos_bolt",
           "within_shape"]
PRIMARY = "lib_shape_cond"
OUT = ROOT / "reports" / "library"
WORK = ROOT / "data" / "library"
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
STYLE = {"lib_shape": ("#E69F00", "-"), "lib_shape_cond": ("#D55E00", "-"), "lib_median": ("#6b6b6b", ":"),
         "own_linear": ("#0072B2", "--"), "own_powerlaw": ("#56B4E9", "--"), "chronos2": ("#CC79A7", "-."),
         "chronos_bolt": ("#AA4499", ":"), "within_shape": ("#009E73", "-")}


def run_chronos(Q, curves):
    ctx = WORK / "chronos" / "context.csv"
    ctx.parent.mkdir(parents=True, exist_ok=True)
    z.chronos_context(Q, curves).to_csv(ctx, index=False)
    out = {}
    for name, model in MODELS.items():
        f = WORK / "chronos" / f"{name}.csv"
        if not f.exists():
            subprocess.run([str(CHRONOS_PY), str(ROOT / "scripts" / "chronos_forecast.py"), "--context", str(ctx),
                            "--out", str(f), "--model", model, "--steps", str(z.PRED_STEPS)], check=True)
        out[name] = pd.read_csv(f).set_index(["id"])
    return out


def stratified_ci(F, a, b, n_boot=5000, seed=0):
    """Mean over datasets of per-dataset MAE(a) - MAE(b), cells resampled within each dataset."""
    G = F.dropna(subset=[a, b])
    per = {ds: ((D[a] - D["actual"]).abs().groupby(D["id"]).mean().to_numpy() * 100,
                (D[b] - D["actual"]).abs().groupby(D["id"]).mean().to_numpy() * 100) for ds, D in G.groupby("dataset")}
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for ea, eb in per.values():
        idx = rng.integers(0, len(ea), (n_boot, len(ea)))
        boots += (ea[idx].mean(1) - eb[idx].mean(1)) / len(per)
    diff = np.mean([ea.mean() - eb.mean() for ea, eb in per.values()])
    return {"diff": diff, "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5), "datasets": len(per)}


def fig_examples(F, curves, path):
    dss = sorted(F["dataset"].unique())
    fig, axes = plt.subplots(1, len(dss), figsize=(180 * mm, 48 * mm), layout="constrained")
    for ax, ds in zip(axes, dss):
        D = F[F["dataset"] == ds]
        e = (D[PRIMARY] - D["actual"]).abs().groupby(D["id"]).mean().dropna().sort_values()
        if e.empty:
            continue
        q = e.index[len(e) // 2]  # median-error cell
        g = D[D["id"] == q]
        t, s = curves[q]
        t90 = g["t90"].iloc[0]
        ax.plot(t[t <= t90], s[t <= t90] * 100, "k.", ms=1.5)
        ax.plot(t[(t > t90) & (t <= 3 * t90)], s[(t > t90) & (t <= 3 * t90)] * 100, ".", color="#9a9a9a", ms=1.5)
        ax.fill_between(g["efc"], g[f"{PRIMARY}_lo"] * 100, g[f"{PRIMARY}_hi"] * 100, color="#D55E00", alpha=0.15, lw=0)
        for m in (PRIMARY, "own_linear", "chronos2", "within_shape"):
            if m in g:
                c, ls = STYLE[m]
                ax.plot(g["efc"], g[m] * 100, color=c, ls=ls, lw=0.9, label=m)
        ax.axvline(t90, color="k", lw=0.4, ls=":")
        ax.set_title(f"{ds}\n{q.split('__')[1][:14]}", fontsize=5.5)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("EFC", fontsize=6)
    axes[0].set_ylabel("SOH (%)", fontsize=6)
    axes[-1].legend(fontsize=4.5, frameon=False)
    fig.suptitle("Seen up to 90 % SOH (black), actual future (grey); median-error cell of each dataset", fontsize=7,
                 x=0.01, ha="left")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    meta, curves = z.load_library()
    Q = z.cell_table(meta, curves)
    Q = Q[(Q["n_seen"] >= 4) & (Q["n_future"] >= 1)]
    print(f"{len(Q)} query cells:", Q.groupby("dataset").size().to_dict(), flush=True)
    chron = run_chronos(Q, curves)
    for prefix in ("llm", "haiku"):  # LLM forecasts (llm_trial/zero_shot_llm.py): Opus 5.5, Haiku 5.5
        for cond in ("A", "B", "C"):
            f = WORK / "llm" / f"{prefix}_{cond}.csv"
            name = f"{prefix}_{cond}"
            if f.exists():
                chron[name] = pd.read_csv(f).set_index(["id"])
                if name not in METHODS:
                    METHODS.append(name)
                    STYLE[name] = ({"A": "#332288", "B": "#882255", "C": "#117733"}[cond],
                                   "-" if prefix == "llm" else ":")
    F = z.forecast(meta, curves, chron)
    # combination rule, fixed before the LLM results were seen: equal-weight mean of the medians of the
    # three zero-shot methods that tied (lib_shape, chronos2, own_linear), and the same plus llm_B
    base = ["lib_shape", "chronos2", "own_linear"]
    F["ensemble"] = F[base].mean(1, skipna=False)
    if "llm_B" in F:
        F["ensemble_llm"] = F[base + ["llm_B"]].mean(1, skipna=False)
    for m in ("ensemble", "ensemble_llm"):
        if m in F and m not in METHODS:
            METHODS.append(m)
            STYLE[m] = ("#000000", "-")
    F.to_csv(WORK / "zero_shot_forecasts.csv", index=False)
    pd.set_option("display.width", 250)
    # main comparison: every method incl. the within-dataset upper bound (datasets with >= 11 cells)
    P, O = z.score(F, METHODS)
    P.to_csv(OUT / "zero_shot_by_dataset.csv", index=False)
    O.to_csv(OUT / "zero_shot_overall.csv")
    print("\nmean over datasets (common points):\n" + O.round(2).to_string())
    print("\nper dataset, MAE (SOH points):\n" + P.pivot(index="dataset", columns="method", values="MAE")[
        [m for m in METHODS if m in set(P["method"])]].round(2).to_string())
    cis = []
    for b in METHODS:
        if b != PRIMARY and b in F:
            r = stratified_ci(F, PRIMARY, b)  # on the points both methods forecast
            cis.append({"vs": b, **r})
            print(f"  {PRIMARY} - {b}: {r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] ({r['datasets']} datasets)")
    # secondary, chosen after seeing the table above: the best zero-shot methods against each other
    for a_, b_ in [("lib_shape", "chronos2"), ("lib_shape", "own_linear"), ("chronos2", "own_linear"),
                   ("lib_shape", "within_shape"), ("chronos2", "within_shape")]:
        r = stratified_ci(F, a_, b_)
        cis.append({"vs": f"{a_} - {b_} (secondary, post hoc)", **r})
        print(f"  {a_} - {b_} (post hoc): {r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")
    # LLM comparisons, fixed before the LLM results were seen
    llm_pairs = [("llm_A", "lib_shape"),   # does the LLM read the analogs better than their weighted mean?
                 ("llm_B", "llm_A"),       # does chemistry, conditions and mechanism knowledge help?
                 ("llm_B", "llm_C"),       # do the analogs help, given that knowledge?
                 ("llm_B", "own_linear"), ("llm_B", "chronos2"), ("llm_C", "own_linear"),
                 ("ensemble_llm", "ensemble"), ("ensemble", "own_linear"),
                 # Haiku 5.5, fixed before its results were seen: does a small model keep the gain?
                 ("haiku_B", "llm_B"), ("haiku_C", "llm_C"),
                 ("haiku_A", "lib_shape"), ("haiku_B", "haiku_A"), ("haiku_B", "haiku_C"),
                 ("haiku_B", "own_linear"), ("haiku_C", "own_linear")]
    for a_, b_ in llm_pairs:
        if a_ in F and b_ in F:
            r = stratified_ci(F, a_, b_)
            cis.append({"vs": f"{a_} - {b_} (LLM, pre-registered)", **r})
            print(f"  {a_} - {b_} (LLM, pre-registered): {r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")
    pd.DataFrame(cis).to_csv(OUT / "zero_shot_ci.csv", index=False)
    fig_examples(F, curves, OUT / "figs" / "zero_shot_examples.png")



if __name__ == "__main__":
    main()
