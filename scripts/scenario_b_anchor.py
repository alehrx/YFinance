"""Scenario B+, third round: anchoring the forecast to the use cell's own crossing of the stop level
(knee_analogs/scenario_b.py, curve_from_shape_anchored; fixed before this was run).

    python scripts/scenario_b_anchor.py

Primary: slope_accel_anch vs slope_accel, use_database and early_linear at the 98 % and 95 % stops.
97 % and 96 % are scored too, only for the trade-off curve. Writes reports/library/scenario_b_anchor.csv
(MAE per type, level and method) and scenario_b_anchor_ci.csv; caches the per-cell forecasts in
data/library/scenario_b_early/ (gitignored) for scripts/fig_scenario_b_predictions.py --replot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from knee_analogs import scenario_b as b  # noqa: E402
from knee_analogs import zero_shot as z  # noqa: E402
from scenario_b_study import stratified_ci  # noqa: E402

OUT = ROOT / "reports" / "library"
CACHE = ROOT / "data" / "library" / "scenario_b_early"
LEVELS = (0.98, 0.97, 0.96, 0.95)
BASE = ["slope_accel", "slope_library", "early_accel", "early_library", "oracle_speed"]
METHODS = [m for x in BASE for m in (x, f"{x}_anch")] + ["early_linear", "use_database"]
PRIMARY = [("slope_accel_anch", "slope_accel"), ("slope_accel_anch", "use_database"),
           ("slope_accel_anch", "early_linear")]
SECONDARY = [("slope_library_anch", "slope_library"), ("early_accel_anch", "early_accel"),
             ("early_library_anch", "early_library"), ("oracle_speed_anch", "oracle_speed"),
             ("slope_library_anch", "use_database"), ("slope_accel_anch", "slope_library_anch")]


def main():
    meta, curves = z.load_library()
    CACHE.mkdir(parents=True, exist_ok=True)
    res, cis = [], []
    pd.set_option("display.width", 220)
    for level in LEVELS:
        P = b.forecast_early(meta, curves, level)
        P.to_csv(CACHE / f"forecast_{level:.2f}.csv.gz", index=False)
        T, O = b.score(P, METHODS)
        print(f"\n=== stop at {level:.0%}; MAE (SOH points), common points\n" + T.round(2).to_string()
              + "\nmean over types:\n" + O.round(2).to_string(), flush=True)
        res.append(T.assign(level=level).reset_index())
        for a, c in PRIMARY + SECONDARY:
            r = stratified_ci(P, a, c)
            cis.append({"level": level, "primary": (a, c) in PRIMARY and level in (0.98, 0.95), "a": a, "b": c,
                        **r})
            print(f"  {a} - {c}: {r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]", flush=True)
    pd.concat(res).to_csv(OUT / "scenario_b_anchor.csv", index=False)
    pd.DataFrame(cis).to_csv(OUT / "scenario_b_anchor_ci.csv", index=False)


if __name__ == "__main__":
    main()
