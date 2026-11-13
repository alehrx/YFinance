"""Leave-one-cell-out evaluation of knee prediction: regression vs nearest-neighbour analogs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

from .core import curve_vector, early_features, eol_cycle, knee_point, spec_of

PROTOCOL = ["C1", "Q1", "C2", "C_avg"]


def protocol_cols(T: pd.DataFrame) -> list[str]:
    """Protocol descriptors present in a feature table: the Severson charge policy and/or the
    adapter's ``proto_*`` columns."""
    return [c for c in PROTOCOL if c in T] + [c for c in T.columns if c.startswith("proto_")]


def cell_table(cells: dict, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows, curves = [], []
    for name, d in cells.items():
        f = early_features(d, n)
        if not f:
            continue
        kn, s1, s2 = knee_point(d)
        rows.append({"cell": name, "knee": kn, "eol": eol_cycle(d), "condition": d.attrs.get("condition", ""), **f})
        curves.append(curve_vector(d, n))
    return pd.DataFrame(rows).set_index("cell"), np.array(curves)


def _scale(train: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    med = train.median()
    iqr = (train.quantile(0.75) - train.quantile(0.25)).replace(0, 1).fillna(1)
    return ((X - med) / iqr).fillna(0).clip(-6, 6).to_numpy()


def _knn(dist: np.ndarray, y: np.ndarray, k: int):
    idx = np.argsort(dist)[:k]
    w = 1 / (dist[idx] + 1e-6)
    w /= w.sum()
    return float(np.exp(w @ np.log(y[idx]))), idx


NON_FEATURES = ("knee", "eol", "condition")


def loco_knee(T: pd.DataFrame, curves: np.ndarray, k: int = 5, target: str = "knee",
              leave_out: str = "cell") -> pd.DataFrame:
    """Predict each cell's knee (or EOL) cycle from the other cells only.

    ``leave_out="condition"`` removes every cell of the query's condition from the training
    cells (scaling, regression and analogs), to test transfer to an unseen condition."""
    feats = [c for c in T.columns if c not in NON_FEATURES]
    y = T[target].to_numpy(float)
    groups = T["condition"].to_numpy() if leave_out == "condition" else T.index.to_numpy()
    out = []
    for i, cell in enumerate(T.index):
        tr = groups != groups[i]
        Ttr, ytr = T.iloc[tr], y[tr]
        r = {"cell": cell, "condition": T["condition"].iloc[i], "actual": y[i],
             "mean": float(np.exp(np.log(ytr).mean()))}
        # average of the other cells run under the same condition (protocol known in advance);
        # NaN when the query is the only cell of its condition, or the condition is left out
        same = Ttr["condition"].to_numpy() == T["condition"].iloc[i]
        r["condition_mean"] = float(np.exp(np.log(ytr[same]).mean())) if same.any() else np.nan
        for name, cols in [("protocol_regression", protocol_cols(T)), ("early_regression", feats)]:
            if not cols:  # no protocol information: the regression reduces to the average
                r[name] = r["mean"]
                continue
            Xtr = _scale(Ttr[cols], Ttr[cols])
            Xte = _scale(Ttr[cols], T.iloc[[i]][cols])
            m = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Xtr, np.log(ytr))
            r[name] = float(np.exp(m.predict(Xte)[0]))
        Xtr = _scale(Ttr[feats], Ttr[feats])
        xte = _scale(Ttr[feats], T.iloc[[i]][feats])[0]
        d_feat = np.sqrt(((Xtr - xte) ** 2).mean(1))
        d_curve = np.sqrt(((curves[tr] - curves[i]) ** 2).mean(1))
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
        for name, dist in [("analogs_features", d_feat), ("analogs_curve", d_curve),
                           ("analogs_combined", z(d_feat) + z(d_curve) - (z(d_feat) + z(d_curve)).min())]:
            pred, idx = _knn(dist, ytr, k)
            r[name] = pred
            r[f"{name}_lo"], r[f"{name}_hi"] = ytr[idx].min(), ytr[idx].max()
            r[f"{name}_neighbors"] = ",".join(Ttr.index[idx])
        out.append(r)
    return pd.DataFrame(out).set_index("cell")


METHODS = ["mean", "condition_mean", "protocol_regression", "early_regression", "analogs_features", "analogs_curve", "analogs_combined"]


def summarize(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in [m for m in METHODS if m in P and P[m].notna().any()]:
        e = P[m] - P["actual"]
        r = {"method": m, "n": int(e.notna().sum()), "MAE_cycles": e.abs().mean(), "median_AE": e.abs().median(),
             "MAPE_%": (e.abs() / P["actual"]).mean() * 100}
        if f"{m}_lo" in P:
            r["range_covers_actual_%"] = ((P[f"{m}_lo"] <= P["actual"]) & (P["actual"] <= P[f"{m}_hi"])).mean() * 100
        rows.append(r)
    return pd.DataFrame(rows).set_index("method")


# ------------------------------------------------------------------ early warning
STATE = ["fade", "slope_50", "accel", "IR_delta", "Tavg_delta", "C_avg", "C1", "C2"]


def snapshots(cells: dict, step: int | None = None, start: int | None = None,
              window: int | None = None) -> pd.DataFrame:
    """State of each cell every ``step`` cycles up to its knee: how much capacity it has
    lost, how fast it is fading now (over the last ``window`` cycles), whether fade is
    accelerating; plus cycles to the knee. Defaults come from the cells' ``DatasetSpec``.
    For sparse data a snapshot is taken only at cycles with a check-up, and a fade rate
    needs 2 check-ups in the window instead of 20 cycles. The column is still called
    ``slope_50`` whatever the window."""
    from .core import protocol_features

    rows = []
    for name, d in cells.items():
        spec = spec_of(d)
        st = step or spec.snapshot_step
        s0_ = start or spec.snapshot_start
        win = window or spec.state_window
        need = 2 if spec.sparse else 20
        kn, _, _ = knee_point(d)
        if not np.isfinite(kn):
            continue
        pol = protocol_features(d)
        x, y = d["cycle"].to_numpy(float), d["QDs"].to_numpy(float)
        first = (x >= 2) & (x <= 10)
        q0 = np.median(y[first]) if first.any() else y[0]
        ms = x[(x >= s0_) & (x <= kn)] if spec.sparse else range(s0_, int(kn) + 1, st)
        for m in ms:
            w1 = (x > m - win) & (x <= m)
            w0 = (x > m - 2 * win) & (x <= m - win)
            if w1.sum() < need:
                continue
            s1 = np.polyfit(x[w1], y[w1], 1)[0] * 1000
            s0 = np.polyfit(x[w0], y[w0], 1)[0] * 1000 if w0.sum() >= need else s1
            r = {"cell": name, "cycle": m, "condition": d.attrs.get("condition", ""),
                 "fade": (np.median(y[w1][-10:]) - q0) * 1000,
                 "slope_50": s1, "accel": s1 - s0, "to_knee": kn - m, **pol}
            for col in ("IR", "Tavg"):
                if col in d:
                    v = d[col].replace(0, np.nan)
                    r[f"{col}_delta"] = v[w1].median() - v.iloc[:10].median()
            rows.append(r)
    return pd.DataFrame(rows)


def _at(S: pd.DataFrame, m: float, window: float, sparse: bool) -> pd.DataFrame:
    """The snapshot at cycle ``m``; for sparse check-ups, the latest one within ``window``
    cycles before ``m``."""
    q = S[S["cycle"] == m]
    if not sparse or len(q):
        return q
    q = S[(S["cycle"] <= m) & (S["cycle"] > m - window)]
    return q.iloc[[-1]] if len(q) else q


def loco_warning(S: pd.DataFrame, checkpoints=(100, 150, 200, 250, 300, 350), k: int = 15,
                 window: int = 50, leave_out: str = "cell", sparse: bool = False) -> pd.DataFrame:
    """At each checkpoint, predict cycles left until the knee and P(knee within ``window``).
    ``leave_out="condition"`` also removes the query's whole condition from the memory."""
    STATE_ = [c for c in STATE + [c for c in S.columns if c.startswith("proto_")] if c in S and S[c].notna().any()]
    out = []
    for cell in S["cell"].unique():
        te = S[S["cell"] == cell]
        if leave_out == "condition":
            tr = S[S["condition"] != te["condition"].iloc[0]]
        else:
            tr = S[S["cell"] != cell]
        Xtr = _scale(tr[STATE_], tr[STATE_])
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Xtr, tr["to_knee"])
        for m in checkpoints:
            q = _at(te, m, window, sparse)
            if q.empty:
                continue
            xq = _scale(tr[STATE_], q[STATE_])[0]
            dist = np.sqrt(((Xtr - xq) ** 2).mean(1))
            # at most 2 snapshots per other cell, so one look-alike cell cannot dominate
            order = np.argsort(dist)
            picked, per = [], {}
            for j in order:
                c = tr["cell"].iloc[j]
                if per.get(c, 0) < 2:
                    picked.append(j); per[c] = per.get(c, 0) + 1
                if len(picked) == k:
                    break
            nb = tr.iloc[picked]["to_knee"].to_numpy()
            at_m = (pd.concat([_at(g, m, window, True) for _, g in tr.groupby("cell")])
                    if sparse else tr[tr["cycle"] == m])
            same_cycle = at_m["to_knee"]
            # same, but only cells of the query's condition (protocol known in advance)
            same_cond = at_m.loc[at_m["condition"] == te["condition"].iloc[0], "to_knee"] if len(at_m) else at_m
            out.append({"cell": cell, "cycle": m, "actual": float(q["to_knee"].iloc[0]),
                        "analogs": float(np.median(nb)), "analogs_p_soon": float((nb <= window).mean()),
                        "regression": float(ridge.predict(xq[None])[0]),
                        "cycle_count_only": float(same_cycle.median()) if len(same_cycle) else np.nan,
                        "cycle_count_p_soon": float((same_cycle <= window).mean()) if len(same_cycle) else np.nan,
                        "condition_cycle_count": float(same_cond.median()) if len(same_cond) else np.nan,
                        "condition_cycle_count_p_soon": float((same_cond <= window).mean()) if len(same_cond) else np.nan})
    W = pd.DataFrame(out)
    return W


def warning_summary(W: pd.DataFrame, window: int = 50) -> dict:
    soon = (W["actual"] <= window).astype(int)
    res = {"n_checks": len(W), "share_knee_within_window": soon.mean()}
    for m in ("analogs", "regression", "cycle_count_only", "condition_cycle_count"):
        if m in W and W[m].notna().any():
            res[f"MAE_{m}"] = (W[m] - W["actual"]).abs().mean()
    res["AUC_soon_analogs"] = roc_auc_score(soon, W["analogs_p_soon"])
    res["AUC_soon_regression"] = roc_auc_score(soon, -W["regression"])
    ok = W["cycle_count_p_soon"].notna()
    res["AUC_soon_cycle_count_only"] = roc_auc_score(soon[ok], W.loc[ok, "cycle_count_p_soon"])
    if "condition_cycle_count_p_soon" in W and W["condition_cycle_count_p_soon"].notna().any():
        ok = W["condition_cycle_count_p_soon"].notna()
        res["n_condition_cycle_count"] = int(ok.sum())
        if soon[ok].nunique() == 2:
            res["AUC_soon_condition_cycle_count"] = roc_auc_score(soon[ok], W.loc[ok, "condition_cycle_count_p_soon"])
            # the other methods on the same checks, for a like-for-like comparison
            for m, col in (("analogs", "analogs_p_soon"), ("regression", "regression")):
                sc = W.loc[ok, col] if m == "analogs" else -W.loc[ok, col]
                res[f"AUC_soon_{m}_same_checks"] = roc_auc_score(soon[ok], sc)
                res[f"MAE_{m}_same_checks"] = (W.loc[ok, m] - W.loc[ok, "actual"]).abs().mean()
    return res


# ------------------------------------------------------------------ statistics for the write-up
def main_group(actual: pd.Series) -> pd.Series:
    """Boolean mask of the 'typical' cells: knee inside Tukey's fences (Q1 - 1.5 IQR,
    Q3 + 1.5 IQR) of all knees. Fixed rule, defined on the truth only, never on predictions."""
    q1, q3 = actual.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (actual >= q1 - 1.5 * iqr) & (actual <= q3 + 1.5 * iqr)


def bootstrap_mae_diff(P: pd.DataFrame, a: str, b: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """MAE(a) - MAE(b) over cells, with a 95% bootstrap CI (resampling cells).
    Negative means ``a`` is better."""
    P = P.dropna(subset=[a, b])
    ea = (P[a] - P["actual"]).abs().to_numpy()
    eb = (P[b] - P["actual"]).abs().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), (n_boot, len(ea)))
    boots = ea[idx].mean(1) - eb[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(ea.mean() - eb.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def rank_corr(P: pd.DataFrame, m: str, mask: pd.Series | None = None) -> float:
    from scipy.stats import spearmanr

    Q = P if mask is None else P[mask]
    return float(spearmanr(Q[m], Q["actual"]).statistic) if len(Q) > 2 else np.nan


def rank_corr_within_condition(P: pd.DataFrame, m: str, min_cells: int = 2) -> float:
    """Does the method rank cells *within* a condition? Spearman correlation, pooled over
    conditions with at least ``min_cells`` cells, between predicted and actual values after
    subtracting each condition's mean actual value from both (the same constant per condition, so
    differences between conditions do not count)."""
    from scipy.stats import spearmanr

    if "condition" not in P:
        return np.nan
    Q = P[P.groupby("condition")["actual"].transform("size") >= min_cells].dropna(subset=[m])
    if len(Q) < 3 or Q["condition"].nunique() == len(Q):
        return np.nan
    centre = Q.groupby("condition")["actual"].transform("mean")
    return float(spearmanr(Q[m] - centre, Q["actual"] - centre).statistic)


def summarize_detail(P: pd.DataFrame, methods=None) -> pd.DataFrame:
    """Per method: MAE (cycles, %) overall, on the main group and on the outliers, range
    coverage, rank correlation within the main group; plus the bootstrap CI of
    analogs minus regression."""
    methods = methods or [m for m in METHODS if m in P and P[m].notna().any()]
    typ = main_group(P["actual"])
    rows = []
    for m in methods:
        e = (P[m] - P["actual"]).abs()
        r = {"method": m, "n": int(e.notna().sum()), "MAE_cycles": e.mean(), "MAPE_%": (e / P["actual"]).mean() * 100,
             "MAE_main_group": e[typ].mean(), "MAE_outliers": e[~typ].mean() if (~typ).any() else np.nan,
             "n_main": int(typ.sum()), "n_outliers": int((~typ).sum()),
             # leave-one-out average is mechanically anti-correlated with the truth (-1): not reported
             "rank_corr_main_group": np.nan if m == "mean" else rank_corr(P, m, typ),
             "rank_corr_within_condition": (np.nan if m in ("mean", "condition_mean")
                                            else rank_corr_within_condition(P, m))}
        if f"{m}_lo" in P:
            r["range_covers_actual_%"] = ((P[f"{m}_lo"] <= P["actual"]) & (P["actual"] <= P[f"{m}_hi"])).mean() * 100
        rows.append(r)
    out = pd.DataFrame(rows).set_index("method")
    if {"analogs_features", "early_regression"} <= set(P):
        bd = bootstrap_mae_diff(P, "analogs_features", "early_regression")
        out.attrs["analogs_minus_regression"] = bd
    if {"analogs_features", "condition_mean"} <= set(P) and P["condition_mean"].notna().sum() > 2:
        out.attrs["analogs_minus_condition_mean"] = bootstrap_mae_diff(P, "analogs_features", "condition_mean")
    return out
