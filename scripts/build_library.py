"""Build a cross-dataset library of ageing trajectories in scale-free form (read-only on the raw data).

    python scripts/build_library.py          # -> data/library/cells.csv, data/library/curves/*.csv

Every cell becomes a curve of state of health (SOH = capacity / initial capacity) against equivalent
full cycles (EFC), plus a metadata row: dataset, chemistry, nominal capacity, temperature, charge and
discharge C-rate, depth of discharge, voltage window. Datasets and their capacity series:

| dataset | chemistry | series | time axis |
|---|---|---|---|
| lsd1 | not stated (Ni-based / graphite, 2.4 Ah) | C/5 check-ups (data/lsd1/checkups) | cycles (full depth) |
| mohtat | NMC / graphite pouch, 5 Ah | C/20 check-ups (data/mohtat) | EFC from Ah throughput |
| severson | LFP / graphite 18650, 1.1 Ah | per-cycle discharge capacity | cycles (full depth) |
| stanford_matrix | NCA / graphite+SiOx 21700, 4.84 Ah | 0.2C RPT every 100-200 cycles | EFC from the release |
| stanford_dyn | NCA / SiOx-graphite (EV cell) | per-cycle normalized discharge capacity | cycles (full depth) |
| tongji | NCA 3.5 Ah, NCM 3.5 Ah, NCM+NCA 2.5 Ah 18650 | per-cycle discharge capacity | cycles (full depth) |

Dense per-cycle series are smoothed (rolling median 9, then mean 5) and thinned to <= 600 points;
check-up series are kept as measured. Initial capacity: median of the first 5 smoothed points (dense)
or the first check-up (sparse); Mohtat uses 4.95 Ah (median beginning-of-life C/20 capacity) because
its 45 °C cells have no beginning-of-life check-up.
"""

from __future__ import annotations

import glob
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"C:\Users\alexa\Datasets Batteries")
OUT = ROOT / "data" / "library"


def lsd1():
    idx = pd.read_csv(ROOT / "data" / "lsd1" / "cells_index.csv").set_index("cell")
    for cell, r in idx.iterrows():
        c = pd.read_csv(ROOT / "data" / "lsd1" / "checkups" / f"{cell}.csv")
        meta = dict(chemistry="Ni-based/graphite (not stated)", nominal_ah=2.4, T=25.0, chg_C=r.proto_chg_C,
                    dis_C=r.proto_dis_C, dod=1.0, vmax=r.proto_vmax, vmin=r.proto_vmin, condition=f"P{r.protocol:02d}")
        yield cell, meta, c["cycle"].to_numpy(float), c["QD"].to_numpy(float), True, None


def mohtat():
    for f in sorted(glob.glob(str(ROOT / "data" / "mohtat" / "cell[0-9]*.csv"))):
        d = pd.read_csv(f)
        r = d.iloc[0]
        meta = dict(chemistry="NMC/graphite pouch", nominal_ah=5.0, T=r.proto_T, chg_C=r.proto_chg_C, dis_C=r.proto_dis_C,
                    dod=r.proto_dod, vmax=4.2, vmin=3.0, condition=r.condition)
        yield Path(f).stem, meta, d["cycle"].to_numpy(float), d["QD"].to_numpy(float), True, 4.95


def severson():
    cells = pd.read_csv(RAW / "MIT_Severson2019" / "cycles" / "cells.csv").set_index("cell")
    seen = set()
    for cell, r in cells.iterrows():
        d = pd.read_csv(RAW / "MIT_Severson2019" / "cycles" / f"{cell}.csv")
        q = d["q_dis"].to_numpy(float)
        key = hash(q[:200].round(5).tobytes())
        if key in seen:  # exact duplicate file (e.g. 2017-06-30 cell47 = cell0)
            continue
        seen.add(key)
        meta = dict(chemistry="LFP/graphite", nominal_ah=1.1, T=30.0, chg_C=r.crate_c, dis_C=4.0, dod=1.0, vmax=3.6,
                    vmin=2.0, condition=r.policy)
        yield cell, meta, d["cycle"].to_numpy(float), q, False, None


def stanford_matrix():
    z = zipfile.ZipFile(RAW / "Stanford_agingmatrix" / "raw" / "2025_Aging_Matrix_EES_Submission.zip")
    b = pd.read_csv(z.open([n for n in z.namelist() if n.endswith("bcfm_filled_20240227.csv")][0]),
                    usecols=["seq_num", "equivalent_full_cycles", "rpt_0.2C_discharge_capacity"])
    m = zipfile.ZipFile(RAW / "Stanford_agingmatrix" / "raw" / "metadata.zip")
    p = pd.read_csv(m.open([n for n in m.namelist() if n.endswith("cycling_parameters.csv") and "MACOSX" not in n][0]))
    p = p.set_index("seq_num")
    for s, g in b.groupby("seq_num"):
        g = g.dropna().sort_values("equivalent_full_cycles")
        if s not in p.index or len(g) < 4:
            continue
        r = p.loc[s]
        meta = dict(chemistry="NCA/graphite+SiOx 21700", nominal_ah=4.84, T=25.0, chg_C=float(r.charge_constant_current_1),
                    dis_C=float(r.discharge_constant_current), dod=np.nan, vmax=float(r.charge_cutoff_voltage),
                    vmin=float(r.discharge_cutoff_voltage),
                    condition=f"cc{r.charge_constant_current_1}-{r.charge_constant_current_2}_v{r.charge_cutoff_voltage}"
                              f"_d{r.discharge_constant_current}_v{r.discharge_cutoff_voltage}")
        yield f"seq{int(s)}", meta, g["equivalent_full_cycles"].to_numpy(float), g["rpt_0.2C_discharge_capacity"].to_numpy(float), True, None


def stanford_dyn():
    root = RAW / "Stanford_dyn" / "raw" / "Publishing_data"
    proto = json.load(open(root / "protocol_mapping_dic.json"))
    for f in sorted(glob.glob(str(root / "aging_summary_cell_*.csv"))):
        cell = re.search(r"(cell_\d+)", f).group(1)
        d = pd.read_csv(f)
        name = proto.get(cell, "")
        m = re.search(r"Co(\d+)", name)
        meta = dict(chemistry="NCA/SiOx-graphite (EV cell)", nominal_ah=np.nan, T=35.0, chg_C=0.5,
                    dis_C=1 / float(m.group(1)) if m else np.nan, dod=1.0, vmax=4.2, vmin=3.1,
                    condition=re.sub(r"-I+$", "", name))
        yield cell, meta, d["Cycle"].to_numpy(float), d["Normalized Discharge Capacity [-]"].to_numpy(float), False, None


def tongji():
    for f in sorted(glob.glob(str(RAW / "Tongji_Zhu2022" / "cycles" / "*.csv"))):
        n = Path(f).stem
        m = re.search(r"Dataset_(\d)_(\w+?)__CY(\d+)-(\d+)_(\d+)-#?n?(\d+)", n)
        if not m:
            continue
        ds, chem, T, chg, dis, _ = m.groups()
        chg_C = float("0." + chg[1:]) if chg.startswith("0") else float(chg)
        d = pd.read_csv(f)
        meta = dict(chemistry={"NCA": "NCA 18650", "NCM": "NCM 18650", "NCM_NCA": "NCM+NCA 18650"}[chem],
                    nominal_ah=3.5 if ds in "12" else 2.5, T=float(T), chg_C=chg_C, dis_C=float(dis), dod=1.0,
                    vmax=4.2, vmin=2.65 if ds in "12" else 2.5, condition=f"D{ds}_T{T}_c{chg_C}_d{dis}")
        yield n, meta, d["cycle"].to_numpy(float), d["q_dis_mAh"].to_numpy(float) / 1000, False, None


SOURCES = {"lsd1": lsd1, "mohtat": mohtat, "severson": severson, "stanford_matrix": stanford_matrix,
           "stanford_dyn": stanford_dyn, "tongji": tongji}


def to_soh(t, q, sparse, q_ref):
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[ok], q[ok]
    o = np.argsort(t, kind="stable")
    t, q = t[o], q[o]
    if not sparse:
        s = pd.Series(q)
        med = s.rolling(9, center=True, min_periods=3).median()
        keep = (s - med).abs() <= 0.02 * med.median()  # drop single-cycle glitches
        t, s = t[keep.to_numpy()], s[keep].reset_index(drop=True)
        q = s.rolling(9, center=True, min_periods=3).median().rolling(5, center=True, min_periods=2).mean().to_numpy()
        q0 = float(np.median(q[1:6]))
        step = max(1, len(t) // 600)
        t, q = t[::step], q[::step]
    else:
        q0 = float(q[0])
    q0 = q_ref or q0
    return t, q / q0, q0


def main():
    (OUT / "curves").mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, fn in SOURCES.items():
        n = 0
        for cell, meta, t, q, sparse, q_ref in fn():
            t, soh, q0 = to_soh(t, q, sparse, q_ref)
            if len(t) < 6 or not (0.9 < soh[0] < 1.1):
                continue
            if ds == "severson" and q0 < 1.04:  # continued from an earlier batch: early life missing
                continue
            cid = f"{ds}__{cell}"
            pd.DataFrame({"efc": t, "soh": soh}).to_csv(OUT / "curves" / f"{cid}.csv", index=False)
            rows.append({"id": cid, "dataset": ds, "cell": cell, **meta, "q_initial": q0, "sparse": sparse,
                         "points": len(t), "efc_last": float(t[-1]), "soh_last": float(soh[-1])})
            n += 1
        print(ds, n, "cells", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "cells.csv", index=False)


if __name__ == "__main__":
    main()
