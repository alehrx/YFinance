"""Voltage-curve features for knee prediction, from ``<data>/curves/<cell>.npz`` written by a dataset
adapter (e.g. scripts/convert_lsd.py). Every feature at ``n`` uses only cycles <= n.

The npz holds, on a common voltage grid ``v_grid`` (V):
* ``age_cycles``, ``age_q``: discharge capacity Q(V) of every ageing cycle;
* ``cu_cycles``, ``cu_q``: Q(V) of every slow check-up discharge;
* ``cu_raw`` (V, Q, T rows) with ``cu_offsets``: the raw check-up discharge records.

Feature groups:
* ``dq_age_*``: Severson-style ΔQ(V) = Q_n(V) - Q_ref(V) between the ageing discharge closest to
  cycle n and the one closest to ``ref_cycle``: log10 of the variance, minimum and mean over the
  voltage range both curves cover. At the ageing rate, so partly a resistance signal.
* ``dq_cu_*``: the same between the latest check-up <= n and the first check-up (slow rate, same
  window for every protocol, so closer to a thermodynamic signal).
* ``ica_*``: incremental capacity dQ/dV of the check-ups, latest minus first: height and position
  of the main peak (3.6-3.85 V for the LSD cells), height of the low-voltage peak (3.45-3.6 V),
  and capacity in the high-voltage shoulder (3.85-4.15 V).
* ``dva_*``: differential voltage dV/dQ of the check-ups vs discharged capacity: capacity at the
  dV/dQ valley (``head``) and from the valley to the end of discharge (``tail``), latest minus
  first, relative to the first check-up's capacity. Peak/valley spacings in DVA track the
  capacity of the electrode whose staging dominates that region.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import savgol_filter

MAIN_PEAK = (3.60, 3.85)
LOW_PEAK = (3.45, 3.60)
SHOULDER = (3.85, 4.15)


def load_curves(folder: str, cell: str) -> dict | None:
    p = os.path.join(folder, "curves", f"{cell}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _nearest(cycles: np.ndarray, c: float, upto: float) -> int | None:
    ok = np.flatnonzero(cycles <= upto)
    if not len(ok):
        return None
    return int(ok[np.argmin(np.abs(cycles[ok] - c))])


def _dq_stats(qa: np.ndarray, qb: np.ndarray, prefix: str) -> dict:
    ok = np.isfinite(qa) & np.isfinite(qb)
    if ok.sum() < 20:
        return {}
    dq = qb[ok] - qa[ok]
    return {f"{prefix}_logvar": float(np.log10(np.var(dq) + 1e-12)), f"{prefix}_min": float(dq.min()),
            f"{prefix}_mean": float(dq.mean())}


def ageing_dq_features(z: dict, n: int, ref_cycle: int = 10) -> dict:
    cyc = z["age_cycles"]
    i_ref = _nearest(cyc, ref_cycle, n)
    i_n = _nearest(cyc, n, n)
    if i_ref is None or i_n is None or cyc[i_n] - cyc[i_ref] < 5:
        return {}
    return _dq_stats(z["age_q"][i_ref], z["age_q"][i_n], "dq_age")


def checkup_dq_features(z: dict, n: int) -> dict:
    cyc = z["cu_cycles"]
    idx = np.flatnonzero(cyc <= n)
    if len(idx) < 2:
        return {}
    return _dq_stats(z["cu_q"][idx[0]], z["cu_q"][idx[-1]], "dq_cu")


def _ica(v_grid: np.ndarray, q: np.ndarray):
    ok = np.isfinite(q)
    if ok.sum() < 40:
        return None
    v = v_grid[ok]
    qs = savgol_filter(q[ok], 21, 3)
    return v, -np.gradient(qs, v)  # discharge: Q grows as V falls


def _ica_stats(v, dqdv) -> dict:
    def window(lo, hi):
        m = (v >= lo) & (v <= hi)
        return v[m], dqdv[m]

    vm, im = window(*MAIN_PEAK)
    vl, il = window(*LOW_PEAK)
    vs, is_ = window(*SHOULDER)
    if not (len(vm) and len(vl) and len(vs)):
        return {}
    j = int(np.argmax(im))
    return {"main_h": float(im[j]), "main_v": float(vm[j]), "low_h": float(il.max()),
            "shoulder_q": float(np.trapz(is_, vs))}


def _dva_stats(V: np.ndarray, Q: np.ndarray) -> dict:
    if len(V) < 100 or Q.max() < 0.5:
        return {}
    qg = np.linspace(0, Q.max(), 400)
    order = np.argsort(Q)
    vq = np.interp(qg, Q[order], V[order])
    dvdq = -np.gradient(savgol_filter(vq, 31, 3), qg)
    m = (qg >= 0.4 * qg[-1]) & (qg <= qg[-1] - 0.1 * qg[-1])
    if not m.any():
        return {}
    q_valley = float(qg[m][np.argmin(dvdq[m])])
    return {"head": q_valley, "tail": float(qg[-1] - q_valley), "valley_h": float(dvdq[m].min())}


def ica_dva_features(z: dict, n: int) -> dict:
    cyc = z["cu_cycles"]
    idx = np.flatnonzero(cyc <= n)
    if len(idx) < 2:
        return {}
    i0, i1 = idx[0], idx[-1]
    out = {}
    a, b = _ica(z["v_grid"], z["cu_q"][i0]), _ica(z["v_grid"], z["cu_q"][i1])
    if a and b:
        s0, s1 = _ica_stats(*a), _ica_stats(*b)
        for k in s0.keys() & s1.keys():
            out[f"ica_{k}_delta"] = s1[k] - s0[k]
        if "main_h" in s0:
            out["ica_main_h_rel"] = s1["main_h"] / s0["main_h"] - 1
    off, raw = z["cu_offsets"], z["cu_raw"]
    r0, r1 = raw[off[i0]:off[i0 + 1]], raw[off[i1]:off[i1 + 1]]
    d0, d1 = _dva_stats(r0[:, 0], r0[:, 1]), _dva_stats(r1[:, 0], r1[:, 1])
    if d0 and d1:
        q0 = float(r0[:, 1].max())
        for k in ("head", "tail"):
            out[f"dva_{k}_delta"] = (d1[k] - d0[k]) / q0
        out["dva_valley_h_delta"] = d1["valley_h"] - d0["valley_h"]
    return out


GROUPS = {"dq_age": ageing_dq_features, "dq_cu": checkup_dq_features, "ica_dva": ica_dva_features}


def voltage_features(folder: str, cell: str, n: int, groups=("dq_age", "dq_cu", "ica_dva")) -> dict:
    z = load_curves(folder, cell)
    if z is None:
        return {}
    out = {}
    for g in groups:
        out.update(GROUPS[g](z, n))
    return out
