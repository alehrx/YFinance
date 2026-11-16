"""Scenario B: forecast the whole ageing curve of a cell at *use* conditions that was never measured,
from cells of the same type aged at *accelerated* conditions and observed only down to 90 % SOH.
Everything below was fixed before the first run (no outcome was used to define the split).

Types and split (by test conditions known a priori to speed ageing; middle conditions are left out):

* ``lsd1``: accelerated = charge >= 1.5 C or upper voltage 4.3 V; use = charge 0.5 C and 4.2 V.
* ``mohtat``: accelerated = -5 and 45 degC; use = 25 degC.
* ``tongji_nca`` (Tongji D1) and ``tongji_ncm`` (D2): accelerated = 35 and 45 degC; use = 25 degC.
* ``stanford_matrix``: accelerated = charge >= 0.75 C; use = charge <= 0.3 C.
* Not used: Severson (every cell is fast-charged, 3.6-5.3 C, no mild condition), Stanford dynamic (one
  temperature, low-rate drive cycles, no stress axis), Tongji D3 (all at 25 degC).

Accelerated cells: those that reached 90 % SOH, seen up to and including their first point at or below
90 % (where a test would stop). Use cells: nothing seen. Their whole curve is scored, from the first
point after EFC 0 up to 3 x their true t90 (all points if they never reach 90 %).

Methods (each gives SOH as a function of EFC for every use condition):
* ``accel_as_is``: no acceleration correction. t90 = geometric mean of the accelerated t90s.
* ``speed_regression``: log t90 regressed (ridge, leave-one-out alpha) on 1000 / T(K), log2 charge and
  discharge C-rate, upper and lower voltage and depth of discharge, fitted on the accelerated cells; a
  feature constant in the accelerated cells is dropped (it cannot be extrapolated).
* Both use the same shape: SOH at 0, 0.1, ..., 1.0 t90 = mean of the accelerated cells' normalised
  pre-90 % curves; beyond t90 the mean of their shape-only library forecasts (knee_analogs/zero_shot.py:
  10 nearest pre-90 % shapes from the *other* datasets) up to 3 t90, then the last slope continued.
* ``llm_*``: llm_trial/scenario_b_llm.py (EFC at SOH levels from 97.5 % down to 70 %, as 10 / 50 / 90 %
  quantiles), joined as a table of (condition key, soh, q10, q50, q90).

References (not zero-shot): ``oracle_speed`` = the same shape with the cell's true t90 (isolates the
shape error); ``use_database`` = mean curve of the *other* use cells of the same type (leave-one-out, >= 3
cells with data at that EFC), i.e. what a use-condition database would give.

Score: absolute SOH error (percentage points) per scored point; mean per cell, per type, then over types.
Speed: |log10(t90 predicted / t90 true)| per cell with a t90.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from knee_analogs import zero_shot as z

FEATS = ["invT", "lchg", "ldis", "vmax", "vmin", "dod"]
U_SHAPE = np.r_[np.round(np.arange(0, 1.01, 0.1), 2), np.round(np.arange(1.1, 3.01, 0.1), 2)]


def assign_types(meta: pd.DataFrame) -> pd.DataFrame:
    """type and role ('accelerated' / 'use') per cell; cells outside the split are dropped."""
    m = meta.copy()
    m["type"], m["role"] = None, None

    def put(mask, typ, acc, use):
        m.loc[mask & acc, ["type", "role"]] = [typ, "accelerated"]
        m.loc[mask & use, ["type", "role"]] = [typ, "use"]

    ds = m["dataset"]
    put(ds == "lsd1", "lsd1", (m["chg_C"] >= 1.5) | (m["vmax"] >= 4.3), (m["chg_C"] <= 0.5) & (m["vmax"] <= 4.2))
    put(ds == "mohtat", "mohtat", m["T"].isin([-5.0, 45.0]), m["T"] == 25.0)
    for typ, d in (("tongji_nca", "D1_"), ("tongji_ncm", "D2_")):
        mask = (ds == "tongji") & m["condition"].astype(str).str.startswith(d)
        put(mask, typ, m["T"] >= 35, m["T"] == 25)
    put(ds == "stanford_matrix", "stanford_matrix", m["chg_C"] >= 0.75, m["chg_C"] <= 0.3)
    return m.dropna(subset=["type"])


def features(m: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"invT": 1000 / (m["T"].astype(float) + 273.15), "lchg": np.log2(m["chg_C"].astype(float)),
                         "ldis": np.log2(m["dis_C"].astype(float)), "vmax": m["vmax"].astype(float),
                         "vmin": m["vmin"].astype(float), "dod": m["dod"].astype(float)}, index=m.index)


def cond_key(r) -> str:
    return f"T{r['T']:g}_c{r['chg_C']:g}_d{r['dis_C']:g}_v{r['vmax']:g}-{r['vmin']:g}_dod{r['dod']:g}"


def seen_curve(curves, cid):
    """An accelerated cell as a test stopped at 90 % would see it."""
    t, s = curves[cid]
    t90, t_stop = z.t_at(t, s, with_stop=True)
    keep = t <= t_stop
    return t[keep], s[keep], t90


def shape(meta, curves, acc_ids, lib_Q, lib_T):
    """Mean normalised curve of the accelerated cells on U_SHAPE (beyond 1 from library analogs)."""
    rows = []
    for c in acc_ids:
        t, s, t90 = seen_curve(curves, c)
        pre = np.interp(U_SHAPE[U_SHAPE <= 1] * t90, t, s)
        other = lib_T[lib_T["dataset"] != meta.loc[c, "dataset"]]
        post, _, _, _ = z._analog_forecast(lib_Q, other, curves, z.SHAPE, c, U_SHAPE[U_SHAPE > 1])
        rows.append(np.r_[pre, post])
    S = np.array(rows)
    return np.nanmean(S, 0)


def curve_from_shape(sh, t90, t):
    """SOH at EFC t from a normalised shape and a t90; beyond U_SHAPE[-1] the last slope continues."""
    u = t / t90
    ok = np.isfinite(sh)
    us, ss = U_SHAPE[ok], sh[ok]
    out = np.interp(u, us, ss)
    slope = (ss[-1] - ss[-4]) / (us[-1] - us[-4])
    beyond = u > us[-1]
    out[beyond] = ss[-1] + slope * (u[beyond] - us[-1])
    return out


def curve_from_levels(levels, efc, t):
    """SOH at EFC t from (SOH level, EFC) pairs, through (0, 100 %); beyond the last level the last slope."""
    e = np.r_[0.0, np.maximum.accumulate(np.asarray(efc, float))]
    s = np.r_[1.0, np.asarray(levels, float)]
    out = np.interp(t, e, s)
    beyond = t > e[-1]
    slope = (s[-1] - s[-2]) / max(e[-1] - e[-2], 1e-9)
    out[beyond] = s[-1] + slope * (t[beyond] - e[-1])
    return out


def speed_regression(F_acc, y_acc, F_use):
    from sklearn.linear_model import RidgeCV
    sd = F_acc.std()
    cols = [c for c in FEATS if np.isfinite(sd[c]) and sd[c] > 1e-9]
    if not cols:
        return np.full(len(F_use), y_acc.mean())
    mu, sd = F_acc[cols].mean(), F_acc[cols].std()
    X = ((F_acc[cols] - mu) / sd).to_numpy()
    reg = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, y_acc)
    return reg.predict(((F_use[cols] - mu) / sd).to_numpy())


def forecast(meta, curves, llm: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per scored point of every use cell: actual and each method. Also a per-cell table of t90s."""
    M = assign_types(meta)
    lib_Q = z.cell_table(meta, curves)
    lib_T = lib_Q[lib_Q["n_future"] >= 1]
    F = features(M)
    rows, cells = [], []
    for typ, D in M.groupby("type"):
        acc = [c for c in D.index[D["role"] == "accelerated"] if np.isfinite(z.t_at(*curves[c]))]
        use = list(D.index[D["role"] == "use"])
        assert not set(acc) & set(use)
        sh = shape(meta, curves, acc, lib_Q, lib_T)
        y = np.log([z.t_at(*curves[c]) for c in acc])
        t90_as_is = float(np.exp(y.mean()))
        t90_reg = dict(zip(use, np.exp(speed_regression(F.loc[acc], y, F.loc[use]))))
        for c in use:
            t, s = curves[c]
            t90 = z.t_at(t, s)
            end = 3 * t90 if np.isfinite(t90) else t.max()
            sel = (t > 0) & (t <= end)
            tt, actual = t[sel], s[sel]
            key = cond_key(M.loc[c])
            out = {"accel_as_is": curve_from_shape(sh, t90_as_is, tt),
                   "speed_regression": curve_from_shape(sh, t90_reg[c], tt)}
            if np.isfinite(t90):
                out["oracle_speed"] = curve_from_shape(sh, t90, tt)
            others = [o for o in use if o != c]
            G = np.array([np.interp(tt, *curves[o], left=np.nan, right=np.nan) for o in others])
            n_ok = np.isfinite(G).sum(0)
            with np.errstate(all="ignore"):
                out["use_database"] = np.where(n_ok >= 3, np.nanmean(G, 0), np.nan)
            pred_t90 = {"accel_as_is": t90_as_is, "speed_regression": t90_reg[c]}
            for name, tab in (llm or {}).items():
                L = tab[(tab["type"] == typ) & (tab["key"] == key)].sort_values("soh", ascending=False)
                if len(L):
                    out[name] = curve_from_levels(L["soh"], L["q50"], tt)
                    out[f"{name}_lo"] = curve_from_levels(L["soh"], L["q10"], tt)
                    out[f"{name}_hi"] = curve_from_levels(L["soh"], L["q90"], tt)
                    pred_t90[name] = float(np.interp(0.9, L["soh"][::-1], L["q50"][::-1]))
                    # post hoc (added after seeing the LLM curves): the LLM's t90 with the shared shape
                    out[f"{name}_speed+shape"] = curve_from_shape(sh, pred_t90[name], tt)
            cells.append({"id": c, "type": typ, "key": key, "t90_true": t90, "n_acc": len(acc),
                          **{f"t90_{k}": v for k, v in pred_t90.items()}})
            for j in range(len(tt)):
                rows.append({"id": c, "type": typ, "key": key, "efc": tt[j], "actual": actual[j],
                             **{k: float(v[j]) for k, v in out.items()}})
    return pd.DataFrame(rows), pd.DataFrame(cells)


def score(P: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """MAE (SOH points) per type on the points every method forecasts, and the mean over types."""
    methods = [m for m in methods if m in P and P[m].notna().any()]
    G = P[P[methods].notna().all(1)]
    per = []
    for typ, D in G.groupby("type"):
        r = {"type": typ, "cells": D["id"].nunique()}
        for m in methods:
            r[m] = ((D[m] - D["actual"]).abs() * 100).groupby(D["id"]).mean().mean()
        per.append(r)
    T = pd.DataFrame(per).set_index("type")
    return T, T[methods].mean()


# --- Scenario B+: accelerated tests to 90 % plus a short test of the use cell itself ----------------------
# Fixed before the first run. The use cell is seen up to and including its first point at or below
# LEVEL (0.98 primary, 0.99 secondary), as if a short test at the use condition stopped there. Scored:
# its points after that, up to 3 x its true t90 (cells that reach 90 %).
# Methods (t90 predictions; the curve is the shared shape with that t90, as above):
# * ``early_accel``: t90 = t_LEVEL(use cell) x median over the accelerated cells of t90 / t_LEVEL
#   (their early-to-90 % ratio is assumed to carry over).
# * ``early_library``: the same ratio from every library cell of the *other* datasets (no test of the type).
# * ``early_accel+llm``: geometric mean of the early_accel t90 and the median-of-4-runs Opus (data) t90.
# * ``early_linear``: the use cell's own straight-line trend over (t_LEVEL / 2, t_LEVEL], no shape.
# Compared with speed_regression, the Opus (data, median of runs) curve, oracle_speed and use_database
# on the same points.


def forecast_early(meta, curves, level=0.98, llm_t90: dict | None = None) -> pd.DataFrame:
    M = assign_types(meta)
    lib_Q = z.cell_table(meta, curves)
    lib_T = lib_Q[lib_Q["n_future"] >= 1]
    F = features(M)
    lib_ratio = {}
    for ds in lib_T["dataset"].unique():  # library ratio t90 / t_level per held-out dataset
        r = [z.t_at(*curves[c]) / z.t_at(*curves[c], level=level) for c in lib_T.index[lib_T["dataset"] != ds]]
        lib_ratio[ds] = float(np.nanmedian(r))
    rows = []
    for typ, D in M.groupby("type"):
        acc = [c for c in D.index[D["role"] == "accelerated"] if np.isfinite(z.t_at(*curves[c]))]
        use = list(D.index[D["role"] == "use"])
        sh = shape(meta, curves, acc, lib_Q, lib_T)
        ratio_acc = float(np.nanmedian([seen_curve(curves, c)[2] / z.t_at(*seen_curve(curves, c)[:2], level=level)
                                        for c in acc]))
        y = np.log([z.t_at(*curves[c]) for c in acc])
        t90_reg = dict(zip(use, np.exp(speed_regression(F.loc[acc], y, F.loc[use]))))
        for c in use:
            t, s = curves[c]
            t90 = z.t_at(t, s)
            tx, t_stop = z.t_at(t, s, level=level, with_stop=True)
            if not (np.isfinite(t90) and np.isfinite(tx)):
                continue
            sel = (t > t_stop) & (t <= 3 * t90)
            if not sel.any():
                continue
            tt, actual = t[sel], s[sel]
            seen = t <= t_stop  # everything below uses only these points of the use cell
            ts, ss = t[seen], s[seen]
            tx_seen = z.t_at(ts, ss, level=level)
            pred = {"early_accel": tx_seen * ratio_acc, "early_library": tx_seen * lib_ratio[meta.loc[c, "dataset"]],
                    "speed_regression": t90_reg[c]}
            key = cond_key(M.loc[c])
            if llm_t90 and (typ, key) in llm_t90:
                pred["early_accel+llm"] = float(np.sqrt(pred["early_accel"] * llm_t90[(typ, key)]))
            out = {k: curve_from_shape(sh, v, tt) for k, v in pred.items()}
            m = (ts > tx_seen / 2) & (ts <= t_stop)
            slope = np.polyfit(ts[m], ss[m], 1)[0] if m.sum() >= 2 else (ss[-1] - 1.0) / ts[-1]
            out["early_linear"] = ss[-1] + slope * (tt - ts[-1])
            out["oracle_speed"] = curve_from_shape(sh, t90, tt)
            others = [o for o in use if o != c]
            G = np.array([np.interp(tt, *curves[o], left=np.nan, right=np.nan) for o in others])
            with np.errstate(all="ignore"):
                out["use_database"] = np.where(np.isfinite(G).sum(0) >= 3, np.nanmean(G, 0), np.nan)
            for j in range(len(tt)):
                rows.append({"id": c, "type": typ, "key": key, "efc": tt[j], "actual": actual[j], "t90_true": t90,
                             "t_stop": t_stop, **{f"t90_{k}": v for k, v in pred.items()},
                             **{k: float(v[j]) for k, v in out.items()}})
    return pd.DataFrame(rows)
