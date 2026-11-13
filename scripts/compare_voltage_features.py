"""Test A with voltage-curve features added to the summary features, leave-one-cell-out.

    python scripts/compare_voltage_features.py --data data/lsd1

Feature sets (protocol descriptors are in every set):
  summary          the per-cycle summary features of run_study.py (capacity, fade rate, curvature,
                   temperature, charge time)
  +dq_age          + ΔQ(V) between ageing discharges (cycle n vs cycle 10)
  +dq_cu           + ΔQ(V) between slow check-ups (latest <= n vs first)
  +ica_dva         + dQ/dV and dV/dQ peak changes of the check-ups
  +all_voltage     + all three
  voltage_only     protocol descriptors + all voltage features, no summary features
For each set, the analogs (5 nearest cells) and ridge regression are refit with the same code as
run_study.py. Writes reports/<name>/voltage_features_*.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee_analogs import evaluate as ev  # noqa: E402
from knee_analogs.core import load_cells, spec_of  # noqa: E402
from knee_analogs.voltage_features import GROUPS, voltage_features  # noqa: E402

META = ["knee", "eol", "condition"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "lsd1"))
    args = ap.parse_args()
    data = Path(args.data)
    out = ROOT / "reports" / data.resolve().name
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(str(data))
    spec = spec_of(next(iter(cells.values())))
    rows, preds = [], {}
    for n in spec.early_cycles:
        T, C = ev.cell_table(cells, n)
        V = pd.DataFrame({c: voltage_features(str(data), c, n) for c in T.index}).T
        V.to_csv(out / f"voltage_features_first{n}.csv")
        proto = ev.protocol_cols(T)
        summary = [c for c in T.columns if c not in META]
        groups = {g: [c for c in V.columns if c.startswith(g + "_") or (g == "ica_dva" and c.startswith(("ica_", "dva_")))]
                  for g in GROUPS}
        sets = {"summary": summary}
        for g, cols in groups.items():
            sets[f"+{g}"] = summary + cols
        allv = [c for cols in groups.values() for c in cols]
        sets["+all_voltage"] = summary + allv
        sets["voltage_only"] = proto + allv
        TV = T.join(V)
        base = None
        for name, cols in sets.items():
            P = ev.loco_knee(TV[META + cols], C)
            D = ev.summarize_detail(P, methods=["condition_mean", "early_regression", "analogs_features"])
            preds[(n, name)] = P
            if name == "summary":
                base = P
            r = {"first_n": n, "features": name, "n_features": len(cols),
                 "missing_%": float(TV[cols].isna().mean().mean() * 100)}
            for m, lab in (("analogs_features", "analogs"), ("early_regression", "regression")):
                r[f"MAE_{lab}"] = D.loc[m, "MAE_cycles"]
                r[f"MAPE_{lab}_%"] = D.loc[m, "MAPE_%"]
                r[f"rank_within_condition_{lab}"] = D.loc[m, "rank_corr_within_condition"]
                if name != "summary":
                    Q = P[["actual", m]].join(base[[m]], rsuffix="_base")
                    bd = ev.bootstrap_mae_diff(Q, m, f"{m}_base")
                    r[f"{lab}_vs_summary"] = bd["diff"]
                    r[f"{lab}_vs_summary_ci"] = f"[{bd['ci_lo']:.1f}, {bd['ci_hi']:.1f}]"
            r["range_covers_%"] = D.loc["analogs_features", "range_covers_actual_%"]
            r["MAE_same_condition_average"] = D.loc["condition_mean", "MAE_cycles"]
            rows.append(r)
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "voltage_features_comparison.csv", index=False)
    pd.concat(preds, names=["first_n", "features"]).to_csv(out / "voltage_features_predictions.csv")
    pd.set_option("display.width", 250)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
