"""Leave-one-cell-out evaluation of knee prediction: regression vs nearest-neighbour analogs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

from .core import curve_vector, early_features, eol_cycle, knee_point

PROTOCOL = ["C1", "Q1", "C2", "C_avg"]


def cell_table(cells: dict, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows, curves = [], []
    for name, d in cells.items():
        f = early_features(d, n)
        if not f:
            continue
        kn, s1, s2 = knee_point(d)
        rows.append({"cell": name, "knee": kn, "eol": eol_cycle(d), **f})
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


def loco_knee(T: pd.DataFrame, curves: np.ndarray, k: int = 5, target: str = "knee") -> pd.DataFrame:
    """Predict each cell's knee (or EOL) cycle from the other cells only."""
    feats = [c for c in T.columns if c not in ("knee", "eol")]
    y = T[target].to_numpy(float)
    out = []
    for i, cell in enumerate(T.index):
        tr = np.arange(len(T)) != i
        Ttr, ytr = T.iloc[tr], y[tr]
        r = {"cell": cell, "actual": y[i], "mean": float(np.exp(np.log(ytr).mean()))}
        for name, cols in [("protocol_regression", PROTOCOL), ("early_regression", feats)]:
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


METHODS = ["mean", "protocol_regression", "early_regression", "analogs_features", "analogs_curve", "analogs_combined"]


def summarize(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        e = P[m] - P["actual"]
        r = {"method": m, "MAE_cycles": e.abs().mean(), "median_AE": e.abs().median(),
             "MAPE_%": (e.abs() / P["actual"]).mean() * 100}
        if f"{m}_lo" in P:
            r["range_covers_actual_%"] = ((P[f"{m}_lo"] <= P["actual"]) & (P["actual"] <= P[f"{m}_hi"])).mean() * 100
        rows.append(r)
    return pd.DataFrame(rows).set_index("method")


# ------------------------------------------------------------------ early warning
STATE = ["fade", "slope_50", "accel", "IR_delta", "Tavg_delta", "C_avg", "C1", "C2"]


def snapshots(cells: dict, step: int = 10, start: int = 60) -> pd.DataFrame:
    """State of each cell every ``step`` cycles up to its knee: how much capacity it has
    lost, how fast it is fading now, whether fade is accelerating; plus cycles to the knee."""
    from .core import parse_policy

    rows = []
    for name, d in cells.items():
        kn, _, _ = knee_point(d)
        pol = parse_policy(d.attrs.get("policy", ""))
        x, y = d["cycle"].to_numpy(float), d["QDs"].to_numpy(float)
        q0 = np.median(y[(x >= 2) & (x <= 10)])
        for m in range(start, int(kn) + 1, step):
            w1 = (x > m - 50) & (x <= m)
            w0 = (x > m - 100) & (x <= m - 50)
            if w1.sum() < 20:
                continue
            s1 = np.polyfit(x[w1], y[w1], 1)[0] * 1000
            s0 = np.polyfit(x[w0], y[w0], 1)[0] * 1000 if w0.sum() >= 20 else s1
            r = {"cell": name, "cycle": m, "fade": (np.median(y[w1][-10:]) - q0) * 1000,
                 "slope_50": s1, "accel": s1 - s0, "to_knee": kn - m, **pol}
            for col in ("IR", "Tavg"):
                if col in d:
                    v = d[col].replace(0, np.nan)
                    r[f"{col}_delta"] = v[w1].median() - v.iloc[:10].median()
            rows.append(r)
    return pd.DataFrame(rows)


def loco_warning(S: pd.DataFrame, checkpoints=(100, 150, 200, 250, 300, 350), k: int = 15,
                 window: int = 50) -> pd.DataFrame:
    """At each checkpoint, predict cycles left until the knee and P(knee within ``window``)."""
    out = []
    for cell in S["cell"].unique():
        tr, te = S[S["cell"] != cell], S[S["cell"] == cell]
        Xtr = _scale(tr[STATE], tr[STATE])
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Xtr, tr["to_knee"])
        for m in checkpoints:
            q = te[te["cycle"] == m]
            if q.empty:
                continue
            xq = _scale(tr[STATE], q[STATE])[0]
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
            same_cycle = tr[tr["cycle"] == m]["to_knee"]
            out.append({"cell": cell, "cycle": m, "actual": float(q["to_knee"].iloc[0]),
                        "analogs": float(np.median(nb)), "analogs_p_soon": float((nb <= window).mean()),
                        "regression": float(ridge.predict(xq[None])[0]),
                        "cycle_count_only": float(same_cycle.median()) if len(same_cycle) else np.nan,
                        "cycle_count_p_soon": float((same_cycle <= window).mean()) if len(same_cycle) else np.nan})
    W = pd.DataFrame(out)
    return W


def warning_summary(W: pd.DataFrame, window: int = 50) -> dict:
    soon = (W["actual"] <= window).astype(int)
    res = {"n_checks": len(W), "share_knee_within_window": soon.mean()}
    for m in ("analogs", "regression", "cycle_count_only"):
        res[f"MAE_{m}"] = (W[m] - W["actual"]).abs().mean()
    res["AUC_soon_analogs"] = roc_auc_score(soon, W["analogs_p_soon"])
    res["AUC_soon_regression"] = roc_auc_score(soon, -W["regression"])
    ok = W["cycle_count_p_soon"].notna()
    res["AUC_soon_cycle_count_only"] = roc_auc_score(soon[ok], W.loc[ok, "cycle_count_p_soon"])
    return res
