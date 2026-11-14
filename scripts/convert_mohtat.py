"""Convert the Mohtat et al. (UMich) NMC pouch-cell ageing data to the per-cell layout of
``knee_analogs.core.load_cells``.

    python scripts/convert_mohtat.py                     # -> data/mohtat/

Raw layout (read only): ``Mohtat NMC/data/NN/`` with
* ``cycling_wExpansion.csv``: ageing cycles at the cell's condition (10 s logging);
* ``OCV_wExpansion.csv``: check-ups, one per ``Cycle number`` (every ~41 ageing cycles): C/20 (250 mA)
  CC charge 3.0 -> 4.2 V, 1 h rest, C/20 CC discharge 4.2 -> 3.0 V, with ``Total Ah throughput``;
* ``Crate_wExpansion.csv``, ``Resistance.csv``: rate and resistance tests at the same check-ups (not used).

21 cells, one per condition: chamber temperature -5 / 25 / 45 °C x depth of discharge 0-100 % /
0-50 % x charge / discharge rates (C/5, 1.5C, 2C, drive profile). Nominal 5 Ah.

The 0-50 % cells discharge a fixed ~2.5 Ah per ageing cycle, so their full capacity is only seen at
the check-ups. Therefore the capacity series is the **C/20 check-up discharge capacity** (sparse
points), and the time axis is **equivalent full cycles** (EFC = total Ah throughput / (2 x 5 Ah),
written to the ``cycle`` column) so that half-depth cycles are not counted as full ones.

Writes ``<out>/<cell>.csv`` (cycle = EFC at the check-up, QD = C/20 discharge capacity, QC, ageing
cycle number, Tavg of the preceding ageing cycles, condition, proto_* descriptors), ``curves/<cell>.npz``
(C/20 discharge Q(V) on a voltage grid and the raw discharge records, same format as convert_lsd.py)
and ``dataset.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"C:\Users\alexa\Datasets Batteries\Mohtat NMC\data")
NOMINAL_AH = 5.0
V_GRID = np.round(np.arange(2.0, 4.3001, 0.005), 3)

# chamber temperature of each cell (°C), from the dataset's condition matrix: cells 1-3, 4-6, ... cycle
# through room temperature (25), -5 and 45 °C
TEMPERATURE = {i: (25, -5, 45)[(i - 1) % 3] for i in range(1, 22)}

SPEC = {
    "name": "mohtat",
    "nominal_ah": NOMINAL_AH,
    "eol_fraction": 0.8,
    # 80 % of nominal: the 45 °C cells have no beginning-of-life check-up (first one after 5-29 EFC,
    # already ~4% lower), so "80 % of the first check-up" would be biased for them
    "eol_basis": "nominal",
    "min_start_fraction": None,
    "sparse": True,               # capacity only at C/20 check-ups (7-14 per cell)
    "min_points": 6,
    "early_cycles": [50, 75, 100, 150],   # equivalent full cycles
    "main_n": 100,
    "checkpoints": [50, 100, 150, 200, 250],
    "snapshot_start": 30,
    "state_window": 60,           # >= 2 check-ups at full depth (one every ~39 EFC)
    "warning_window": 50,
    "knee_from": "QD",
    "knee_accelerating_only": True,
    # most cells only bend after 80 % (or not at all); fit the knee over the whole test (to 58-84 %)
    "knee_until_eol": False,
    "exclude_cells": {},
    # no clear knee (scripts/check_knees.py rule, whole trajectory): kept for the end-of-life test
    "no_knee_cells": {
        "cell02": "T-5_dod1.0_ch0.20_dis0.20: no breakpoint with accelerating fade",
        "cell03": "T+45_dod1.0_ch0.20_dis0.20: no breakpoint with accelerating fade",
        "cell05": "T-5_dod1.0_ch1.50_dis1.50: no breakpoint with accelerating fade",
        "cell06": "T+45_dod1.0_ch1.50_dis1.50: fade after/before breakpoint 1.12 < 1.3",
        "cell08": "T-5_dod1.0_ch2.00_dis2.00: no breakpoint with accelerating fade",
        "cell11": "T-5_dod1.0_ch0.20_dis1.50: no breakpoint with accelerating fade",
        "cell14": "T-5_dod0.5_ch0.20_dis0.20: no breakpoint with accelerating fade",
        "cell15": "T+45_dod0.5_ch0.20_dis0.20: no breakpoint with accelerating fade",
        "cell16": "T+25_dod0.5_ch0.20_dis1.50: no breakpoint with accelerating fade",
        "cell17": "T-5_dod0.5_ch0.20_dis1.50: no breakpoint with accelerating fade",
        "cell18": "T+45_dod0.5_ch0.20_dis1.50: no breakpoint with accelerating fade",
        "cell19": "T+25_dod0.5_ch1.50_dis0.30: no breakpoint with accelerating fade",
        "cell20": "T-5_dod0.5_ch1.50_dis0.30: no breakpoint with accelerating fade",
        "cell21": "T+45_dod0.5_ch1.50_dis0.30: no breakpoint with accelerating fade"
    },
    "extra": {"source": "Mohtat NMC/data (UMich pouch, DOI 10.7302/7tw1-kc35)",
              "axis": "cycle column = equivalent full cycles (Ah throughput / 10 Ah)",
              "capacity": "C/20 check-up discharge capacity"},
}


def q_of_v(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v) & np.isfinite(q)
    v, q = v[ok], q[ok]
    if len(v) < 5:
        return np.full(len(V_GRID), np.nan, np.float32)
    vm = np.minimum.accumulate(v)
    keep = np.r_[True, np.diff(vm) < 0]
    return np.interp(V_GRID, vm[keep][::-1], q[keep][::-1], left=np.nan, right=np.nan).astype(np.float32)


def segments(g: pd.DataFrame) -> pd.DataFrame:
    s = np.sign(g["Current [mA]"].round(-1))
    g = g.assign(seg=(s.diff() != 0).cumsum())
    return g


def checkups(cell_dir: Path):
    d = pd.read_csv(cell_dir / "OCV_wExpansion.csv")
    rows, qv, raw = [], [], []
    for c, g in d.groupby("Cycle number", sort=True):
        g = segments(g)
        dis = [x for _, x in g.groupby("seg") if x["Current [mA]"].median() < -200 and x["Current [mA]"].median() > -300
               and len(x) > 100]
        chg = [x for _, x in g.groupby("seg") if 200 < x["Current [mA]"].median() < 300 and len(x) > 100]
        if not dis:
            continue
        x = dis[-1]
        t = x["Time [s]"].to_numpy()
        i = -x["Current [mA]"].to_numpy() / 1000
        q = np.r_[0, np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t)) / 3600]
        qc = np.nan
        if chg:
            y = chg[-1]
            ty, iy = y["Time [s]"].to_numpy(), y["Current [mA]"].to_numpy() / 1000
            qc = float(np.sum(0.5 * (iy[1:] + iy[:-1]) * np.diff(ty)) / 3600)
        rows.append({"ageing_cycle": int(c), "QD": float(q[-1]), "QC": qc,
                     "throughput_ah": float(g["Total Ah throughput [Ah]"].max()),
                     "v_end": float(x["Voltage [V]"].iloc[-1]), "T_checkup": float(g["Temperature [C]"].mean())})
        qv.append(q_of_v(x["Voltage [V]"].to_numpy(), q))
        raw.append(np.column_stack([x["Voltage [V]"].to_numpy(), q, x["Temperature [C]"].to_numpy()]).astype(np.float32))
    return pd.DataFrame(rows), qv, raw


def protocol(cell_dir: Path, idx: int) -> tuple[dict, pd.Series]:
    """Charge / discharge current and depth of discharge, measured on the first 20 ageing cycles;
    mean cell temperature per ageing cycle."""
    d = pd.read_csv(cell_dir / "cycling_wExpansion.csv",
                    usecols=["Time [s]", "Current [mA]", "Voltage [V]", "Temperature [C]", "Cycle number"])
    tavg = d.groupby("Cycle number")["Temperature [C]"].mean()
    e = d[(d["Cycle number"] >= 1) & (d["Cycle number"] <= 20)]
    dt = e["Time [s]"].diff().fillna(0).clip(0, 60).to_numpy()
    i = e["Current [mA]"].to_numpy() / 1000
    chg, dis = i > 0.05, i < -0.05
    dis_ah = float(np.sum(-i[dis] * dt[dis]) / 3600) / e["Cycle number"].nunique()
    # charge rate: the constant-current part (95th percentile of the charge current)
    proto = {"proto_T": float(TEMPERATURE[idx]),
             "proto_chg_C": round(float(np.percentile(i[chg], 95)) / NOMINAL_AH, 2),
             # time-averaged discharge current (drive profiles vary within the cycle)
             "proto_dis_C": round(float(np.sum(-i[dis] * dt[dis]) / np.sum(dt[dis])) / NOMINAL_AH, 2),
             # set voltage window (0-100 % or 0-50 %); the Ah actually moved is lower at -5 °C
             "proto_dod": 1.0 if dis_ah / NOMINAL_AH > 0.6 else 0.5,
             "proto_dis_std_C": round(float(np.std(-i[dis])) / NOMINAL_AH, 2)}
    return proto, tavg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(ROOT / "data" / "mohtat"))
    ap.add_argument("--spec-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    if not args.spec_only:
        index = []
        for cell_dir in sorted(Path(args.raw).iterdir()):
            if not cell_dir.is_dir():
                continue
            idx = int(cell_dir.name)
            C, qv, raw = checkups(cell_dir)
            proto, tavg = protocol(cell_dir, idx)
            C["cycle"] = C["throughput_ah"] / (2 * NOMINAL_AH)
            prev = [0] + C["ageing_cycle"].tolist()[:-1]
            C["Tavg"] = [tavg.loc[(tavg.index > a) & (tavg.index <= b)].mean() if b > a else np.nan
                         for a, b in zip(prev, C["ageing_cycle"])]
            C["condition"] = (f"T{proto['proto_T']:+.0f}_dod{proto['proto_dod']:.1f}_"
                              f"ch{proto['proto_chg_C']:.2f}_dis{proto['proto_dis_C']:.2f}")
            for k, v in proto.items():
                C[k] = v
            name = f"cell{idx:02d}"
            cols = ["cycle", "QD", "QC", "ageing_cycle", "throughput_ah", "Tavg", "T_checkup", "v_end", "condition",
                    *proto]
            C[cols].to_csv(out / f"{name}.csv", index=False)
            lengths = np.array([len(r) for r in raw], np.int64)
            np.savez_compressed(out / "curves" / f"{name}.npz", v_grid=V_GRID.astype(np.float32),
                                age_cycles=np.zeros(0, np.float32), age_q=np.zeros((0, len(V_GRID)), np.float32),
                                cu_cycles=C["cycle"].to_numpy(np.float32), cu_q=np.array(qv, np.float32),
                                cu_raw=np.concatenate(raw), cu_offsets=np.r_[0, np.cumsum(lengths)])
            index.append({"cell": name, "checkups": len(C), "last_efc": round(float(C["cycle"].max()), 1),
                          "q0": round(float(C["QD"].iloc[0]), 3), "q_end_frac": round(float(C["QD"].iloc[-1] / C["QD"].iloc[0]), 3),
                          **proto})
            print(index[-1], flush=True)
        pd.DataFrame(index).to_csv(out / "cells_index.csv", index=False)
    with open(out / "dataset.json", "w", encoding="utf-8") as fh:
        json.dump(dict(SPEC, name=out.name), fh, indent=2)


if __name__ == "__main__":
    main()
