"""Run-to-run spread of the Scenario B LLM forecasts (llm_trial/scenario_b_llm.py --repeat 1..3).
Fixed before the repeats were seen: each model x condition has 4 runs (the original and 3 repeats);
each run is scored like scripts/scenario_b_study.py (same points: cells that reach 90 %). The combined
forecast is the median over the 4 runs of the EFC at each SOH level and quantile.

    python scripts/scenario_b_repeats.py

Writes reports/library/scenario_b_repeats.csv (one row per run and the median), and
scenario_b_repeats_t90.csv (spread of the predicted t90 across runs, per type).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from knee_analogs import scenario_b as b  # noqa: E402
from knee_analogs import zero_shot as z  # noqa: E402
from scenario_b_study import stratified_ci  # noqa: E402

LLM = ROOT / "data" / "library" / "llm" / "scenario_b"
OUT = ROOT / "reports" / "library"
GROUPS = ["opus_data", "opus_know", "haiku_data", "haiku_know"]


def main():
    meta, curves = z.load_library()
    llm = {}
    for g in GROUPS:
        runs = [pd.read_csv(LLM / f"{g}.csv")] + [pd.read_csv(f) for f in sorted((LLM / "repeats").glob(f"{g}_r*.csv"))]
        for i, R in enumerate(runs):
            llm[f"{g}_r{i}"] = R
        A = pd.concat(runs)
        llm[f"{g}_med"] = A.groupby(["type", "key", "soh"], as_index=False)[["q10", "q50", "q90"]].median()
        print(g, len(runs), "runs", flush=True)
    P, C = b.forecast(meta, curves, llm)
    names = list(llm) + [f"{n}_speed+shape" for n in llm]
    T, O = b.score(P, names + ["oracle_speed", "use_database"])
    rows = []
    for n in names:
        g = n.split("_r")[0].split("_med")[0]
        rows.append({"method": n, "group": g, "speed+shape": n.endswith("_speed+shape"),
                     "run": "median" if "_med" in n else n.split("_r")[1].split("_")[0],
                     "MAE": O[n], **{t: T.loc[t, n] for t in T.index}})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "scenario_b_repeats.csv", index=False)
    pd.set_option("display.width", 220)
    single = R[R["run"] != "median"]
    S = single.groupby(["group", "speed+shape"])["MAE"].agg(["mean", "std", "min", "max"])
    S["median_of_runs"] = R[R["run"] == "median"].set_index(["group", "speed+shape"])["MAE"]
    print("\nmean over types, SOH points: single runs (mean, sd, range) and the median-of-runs forecast\n"
          + S.round(2).to_string())
    print("\nper type, single runs: sd across runs\n"
          + single.groupby(["group", "speed+shape"])[list(T.index)].std().round(2).to_string())
    # spread of the predicted t90 across runs: max / min over the 4 runs, median over target conditions
    t90 = []
    for g in GROUPS:
        cols = [f"t90_{g}_r{i}" for i in range(4) if f"t90_{g}_r{i}" in C]
        D = C.drop_duplicates(["type", "key"])
        ratio = D[cols].max(1) / D[cols].min(1)
        t90.append({"group": g, **ratio.groupby(D["type"]).median().round(2).to_dict()})
    t90 = pd.DataFrame(t90).set_index("group")
    t90.to_csv(OUT / "scenario_b_repeats_t90.csv")
    print("\nmax / min predicted t90 over the 4 runs (median over target conditions)\n" + t90.to_string())
    cis = []
    for a, c in [("opus_data_med", "opus_know_med"), ("haiku_data_med", "haiku_know_med"),
                 ("haiku_data_med", "opus_data_med"), ("opus_data_med", "speed_regression"),
                 ("opus_data_med_speed+shape", "opus_data_med"), ("opus_data_med_speed+shape", "use_database"),
                 ("haiku_data_med_speed+shape", "haiku_data_med")]:
        r = stratified_ci(P, a, c)
        cis.append({"a": a, "b": c, **r})
        print(f"  {a} - {c}: {r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")
    pd.DataFrame(cis).to_csv(OUT / "scenario_b_repeats_ci.csv", index=False)


if __name__ == "__main__":
    main()
