"""Scenario B+, second round: a later stop of the short use test (95 % SOH) and slope scaling (the fade rate
at the end of the short test, which skips the break-in). Fixed before it was run; see
knee_analogs/scenario_b.py (forecast_early).

    python scripts/scenario_b_early2.py

Writes reports/library/scenario_b_early2.csv, scenario_b_early2_ci.csv and scenario_b_early2_t90.csv.
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
from scenario_b_early import opus_median  # noqa: E402
from scenario_b_study import stratified_ci  # noqa: E402

OUT = ROOT / "reports" / "library"
METHODS = ["slope_accel", "slope_library", "slope_accel+llm", "early_accel", "early_library", "early_accel+llm",
           "early_linear", "oracle_speed", "use_database"]


def main():
    meta, curves = z.load_library()
    _, llm_t90 = opus_median()
    res, cis, t90s = [], [], []
    pd.set_option("display.width", 220)
    for level in (0.98, 0.95):
        P = b.forecast_early(meta, curves, level, llm_t90)
        T, O = b.score(P, METHODS)
        C = P.drop_duplicates("id")
        print(f"\n=== short use test to {level:.0%} SOH (stop at a median "
              f"{(C['t_stop'] / C['t90_true']).median():.0%} of t90); MAE (SOH points), common points\n"
              + T.round(2).to_string() + "\nmean over types:\n" + O.round(2).to_string())
        print("stop at (median fraction of true t90):", (C["t_stop"] / C["t90_true"]).groupby(C["type"]).median()
              .round(2).to_dict())
        res.append(T.assign(level=level).reset_index())
        for k in [c for c in C if c.startswith("t90_") and c != "t90_true"]:
            f = 10 ** np.abs(np.log10(C[k] / C["t90_true"]))
            t90s.append({"level": level, "method": k[4:], **f.groupby(C["type"]).median().round(2).to_dict()})
        for a, c in [("slope_accel", "early_accel"), ("slope_library", "early_library"),
                     ("slope_accel", "slope_library"), ("slope_accel", "early_linear"),
                     ("slope_accel+llm", "slope_accel"), ("slope_accel", "use_database"),
                     ("early_accel", "use_database")]:
            r = stratified_ci(P, a, c)
            cis.append({"level": level, "a": a, "b": c, **r})
            print(f"  {a} - {c}: {r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")
    pd.concat(res).to_csv(OUT / "scenario_b_early2.csv", index=False)
    pd.DataFrame(cis).to_csv(OUT / "scenario_b_early2_ci.csv", index=False)
    S = pd.DataFrame(t90s)
    S.to_csv(OUT / "scenario_b_early2_t90.csv", index=False)
    print("\nmedian factor between predicted and true t90\n" + S.to_string(index=False))


if __name__ == "__main__":
    main()
