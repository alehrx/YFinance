"""Convert the LSD first-life cycling files (Arbin exports, one CSV per cell, 10 s sampling) to the
per-cell layout of ``knee_analogs.core.load_cells``.

    python scripts/convert_lsd.py                       # -> data/lsd1/
    python scripts/convert_lsd.py --life "LSD 2nd life" --out data/lsd2

The raw files are only read. Each cycle is one CC-CV charge (to 4.2 or 4.3 V, ending at 0.048 A),
a rest, one discharge, a rest. Every 5th cycle (step indices 12-17) is a check-up: C/5 (0.48 A)
discharge to 3.0 V with 30 min rests. All other cycles (step indices 22-25) are ageing cycles at the
protocol's currents. They are kept apart:

* ``<out>/<cell>.csv``: one row per ageing cycle: ``cycle`` (the cycler's cycle index, so
  check-ups count as cycles), ``QD``/``QC`` (Ah), ``Tavg``/``Tmax`` (°C, during the cycle),
  ``chargetime`` (min), ``condition`` (protocol id) and ``proto_*`` descriptors of the protocol, measured
  on the first ageing cycles: charge CC current and discharge current as C-rates of 2.4 Ah, upper and
  lower cut-off voltage, pulsed discharge flag.
* ``<out>/checkups/<cell>.csv``: one row per C/5 check-up (cycle, QD, QC, Tavg, cut-off voltages).
* ``<out>/curves/<cell>.npz`` (numpy, no pickle): discharge Q(V) on a common voltage grid for every
  ageing cycle and every check-up, plus the raw 10 s check-up discharge records (V, Q, T) for
  dQ/dV and DVA.
* ``<out>/dataset.json``: the per-dataset settings (``knee_analogs.core.DatasetSpec``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"C:\Users\alexa\Datasets Batteries")
NOMINAL_AH = 2.4  # protocol currents are multiples of 0.48 A = C/5 of 2.4 Ah
V_GRID = np.round(np.arange(2.0, 4.3001, 0.005), 3)
COLS = ["Test_Time(s)", "Step_Time(s)", "Cycle_Index", "Step_Index", "Current(A)", "Voltage(V)",
        "Discharge_Capacity(Ah)", "Charge_Capacity(Ah)", "Aux_Temperature_1(C)", "Battery_ID", "Protocol_ID"]
CHECKUP_STEPS = {12, 13, 14, 15, 16, 17}
AGEING_STEPS = {22, 23, 24, 25}

SPEC = {
    "name": "lsd1",
    "nominal_ah": NOMINAL_AH,
    "eol_fraction": 0.8,
    # 80% of each cell's own initial capacity: the 4.3 V protocols start ~8% above the 4.2 V ones
    "eol_basis": "initial",
    "min_start_fraction": None,
    "sparse": False,
    "min_points": 40,                 # shortest cells reach end of life after ~66 cycles
    "early_cycles": [25, 50, 100, 200],
    "main_n": 50,                     # before the knee of almost every cell (life 66-1436 cycles)
    "checkpoints": [50, 100, 200, 300, 400, 600, 800, 1000],
    "snapshot_step": 10,
    "snapshot_start": 30,
    "state_window": 25,
    "warning_window": 50,
    # knee and end of life from the C/5 check-ups (same rate and 3.0-4.2 V window for every
    # protocol); at 2-3C the ageing-cycle capacity has rate artefacts and an early drop
    "knee_from": "checkup",
    "knee_accelerating_only": True,
    # no clear knee (scripts/check_knees.py rule: fade after the breakpoint >= 1.3x faster than
    # before, breakpoint between 20% and 95% of end of life), decided on the capacity curves only
    # All six P06 cells (2C charge) fade almost linearly from the first cycle: no knee to predict.
    "exclude_cells": {
        "cell13": "P06: no clear knee (breakpoint at 0.17 of end of life, fade nearly linear)",
        "cell14": "P06: no clear knee (fade after/before breakpoint 1.24 < 1.3)",
        "cell15": "P06: no clear knee (fade after/before breakpoint 1.17 < 1.3)",
        "cell16": "P06: no clear knee (fade after/before breakpoint 1.20 < 1.3)",
        "cell71": "P06: no clear knee (fade after/before breakpoint 1.20 < 1.3)",
        "cell72": "P06: no clear knee (fade after/before breakpoint 1.10 < 1.3)",
        "cell22": "P08: knee not before end of life (breakpoint at 0.97 of end of life)",
        "cell24": "P08: no clear knee (fade after/before breakpoint 1.21 < 1.3)",
        "cell25": "P09: no clear knee (fade after/before breakpoint 1.27 < 1.3)",
        "cell31": "P01: knee not before end of life (breakpoint at 0.97 of end of life)",
        "cell60": "P01: no knee, capacity falls linearly to end of life (ratio 1.02)",
    },
    "extra": {"source": "LSD 1st life (Arbin CSV, 10 s)", "cells": "single cell type, 2.4 Ah implied by currents",
              "check-ups": "C/5 discharge every 5th cycle"},
}


def q_of_v(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Discharge capacity at each grid voltage (NaN outside the measured range). Voltage falls
    during discharge; the running minimum makes it monotonic (relaxation blips, pulses)."""
    ok = np.isfinite(v) & np.isfinite(q)
    v, q = v[ok], q[ok]
    if len(v) < 5:
        return np.full(len(V_GRID), np.nan, np.float32)
    vm = np.minimum.accumulate(v)
    # keep the first sample of each new voltage minimum
    keep = np.r_[True, np.diff(vm) < 0]
    vm, qm = vm[keep][::-1], q[keep][::-1]
    out = np.interp(V_GRID, vm, qm, left=np.nan, right=np.nan)
    return out.astype(np.float32)


def convert_cell(path: str):
    d = pd.read_csv(path, usecols=COLS)[COLS]  # usecols keeps file order; reorder before renaming
    d.columns = ["t", "st", "cycle", "step", "I", "V", "Qd", "Qc", "T", "battery", "protocol"]
    battery, protocol = int(d["battery"].iloc[0]), int(d["protocol"].iloc[0])
    rows, crow, qv_age, qv_cu, cu_raw = [], [], [], [], []
    for c, g in d.groupby("cycle", sort=True):
        steps = set(g["step"].unique())
        dis = g[g["I"] < -1e-3]
        chg = g[g["I"] > 1e-3]
        if dis.empty or chg.empty:
            continue
        base = {"cycle": int(c), "QD": dis["Qd"].max(), "QC": chg["Qc"].max(), "Tavg": g["T"].mean(),
                "Tmax": g["T"].max(), "chargetime": (chg["t"].max() - chg["t"].min()) / 60,
                "vmax": chg["V"].max(), "vmin": dis["V"].min()}
        if steps <= CHECKUP_STEPS:
            crow.append(base)
            qv_cu.append(q_of_v(dis["V"].to_numpy(), dis["Qd"].to_numpy()))
            cu_raw.append(dis[["V", "Qd", "T"]].to_numpy(np.float32))
        elif steps <= AGEING_STEPS:
            base["i_chg_max"] = chg["I"].max()
            base["i_dis"] = -dis["I"].median()
            base["n_dis_steps"] = int((dis["step"].diff().fillna(0) != 0).sum() + 1)
            base["rest_in_dis"] = int(((g["I"].abs() < 1e-3) & (g["t"] > dis["t"].min()) & (g["t"] < dis["t"].max())).sum())
            rows.append(base)
            qv_age.append(q_of_v(dis["V"].to_numpy(), dis["Qd"].to_numpy()))
        # other step patterns (none seen so far) are reported by the caller
    A, C = pd.DataFrame(rows), pd.DataFrame(crow)
    first = A.iloc[1:6] if len(A) > 6 else A
    proto = {
        "proto_chg_C": round(float(first["i_chg_max"].median()) / NOMINAL_AH, 3),
        "proto_dis_C": round(float(first["i_dis"].median()) / NOMINAL_AH, 3),
        "proto_vmax": round(float(first["vmax"].median()), 2),
        "proto_vmin": round(float(first["vmin"].median()), 2),
        "proto_pulsed": int(first["rest_in_dis"].median() > 0),
    }
    A = A.drop(columns=["i_chg_max", "i_dis", "n_dis_steps", "rest_in_dis"])
    A["condition"] = f"P{protocol:02d}"
    for k, v in proto.items():
        A[k] = v
    A["battery_id"] = battery
    other = sorted(set(d["cycle"].unique()) - set(A["cycle"]) - set(C["cycle"] if len(C) else []))
    lengths = np.array([len(x) for x in cu_raw], np.int64)
    curves = dict(v_grid=V_GRID.astype(np.float32),
                  age_cycles=A["cycle"].to_numpy(np.int32), age_q=np.array(qv_age, np.float32),
                  cu_cycles=C["cycle"].to_numpy(np.int32) if len(C) else np.zeros(0, np.int32),
                  cu_q=np.array(qv_cu, np.float32).reshape(len(qv_cu), len(V_GRID)),
                  cu_raw=np.concatenate(cu_raw) if cu_raw else np.zeros((0, 3), np.float32),
                  cu_offsets=np.r_[0, np.cumsum(lengths)])
    return A, C, curves, other, protocol, battery


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--life", default="LSD 1st life")
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(ROOT / "data" / "lsd1"))
    ap.add_argument("--spec-only", action="store_true", help="only rewrite dataset.json")
    args = ap.parse_args()
    out = Path(args.out)
    if args.spec_only:
        write_spec(out)
        return
    (out / "checkups").mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.raw, args.life, "*.csv")),
                   key=lambda f: int(re.sub(r"\D", "", os.path.basename(f)) or 0))
    index = []
    for f in files:
        A, C, curves, other, protocol, battery = convert_cell(f)
        name = f"cell{int(os.path.basename(f)[:-4]):02d}"
        A.to_csv(out / f"{name}.csv", index=False)
        C.to_csv(out / "checkups" / f"{name}.csv", index=False)
        np.savez_compressed(out / "curves" / f"{name}.npz", **curves)
        index.append({"cell": name, "raw_file": os.path.basename(f), "battery_id": battery, "protocol": protocol,
                      "ageing_cycles": len(A), "checkups": len(C), "other_cycles": len(other),
                      "last_cycle": int(A["cycle"].max()), **{k: A[k].iloc[0] for k in A if k.startswith("proto_")}})
        print(index[-1], flush=True)
    pd.DataFrame(index).to_csv(out / "cells_index.csv", index=False)
    write_spec(out)
    print(f"wrote {len(index)} cells to {out}")


def write_spec(out: Path) -> None:
    with open(out / "dataset.json", "w", encoding="utf-8") as fh:
        json.dump(dict(SPEC, name=out.name), fh, indent=2)


if __name__ == "__main__":
    main()
