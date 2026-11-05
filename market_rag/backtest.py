"""Walk-forward evaluation: refit yearly, query every ``step`` trading days, never peek."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .rag import MarketRAG, RAGConfig


def _refit_dates(index: pd.DatetimeIndex, start, end, years: int):
    d = pd.Timestamp(start)
    end = pd.Timestamp(end)
    out = []
    while d <= end:
        out.append(d)
        d = d + pd.DateOffset(years=years)
    return out


def walk_forward(kb: pd.DataFrame, cfg: RAGConfig, test_start, test_end=None, refit_years: int = 1,
                 step: int = 5, require_text: bool = False, verbose: bool = True) -> pd.DataFrame:
    h = cfg.horizon
    test_end = pd.Timestamp(test_end) if test_end is not None else kb.index[-1]
    rows = []
    for r0 in _refit_dates(kb.index, test_start, test_end, refit_years):
        r1 = min(r0 + pd.DateOffset(years=refit_years), test_end + pd.Timedelta(days=1))
        model = MarketRAG(cfg).fit(kb, train_end=r0)
        if verbose:
            print("  ", model.summary())
        qi = np.flatnonzero((kb.index >= r0) & (kb.index < r1))[::step]
        for i in qi:
            if np.isnan(model.y[i]):
                continue
            if require_text and not model.has_text[i]:
                continue
            p = model.predict_index(i)
            if p is None:
                continue
            rows.append(dict(date=kb.index[i], exp_ret=p.exp_ret, p_up=p.p_up, median_ret=p.median_ret,
                             actual=model.y[i], base_rate=model.base_rate, alpha=p.alpha, k=p.k,
                             top_analog=p.neighbors.index[int(np.argmax(p.neighbors["weight"]))]))
    return pd.DataFrame(rows).set_index("date")


def logistic_baseline(kb: pd.DataFrame, cfg: RAGConfig, test_start, test_end=None,
                      refit_years: int = 1, step: int = 5, require_text: bool = False) -> pd.DataFrame:
    """Same schedule, a plain regularised logistic regression on the same metrics."""
    h = cfg.horizon
    test_end = pd.Timestamp(test_end) if test_end is not None else kb.index[-1]
    y = kb[f"fwd_ret_{h}"]
    label_date = pd.to_datetime(kb[f"label_date_{h}"])
    rows = []
    for r0 in _refit_dates(kb.index, test_start, test_end, refit_years):
        r1 = min(r0 + pd.DateOffset(years=refit_years), test_end + pd.Timedelta(days=1))
        tr = (label_date < r0) & y.notna()
        feats = [c for c in cfg.features if c in kb and kb.loc[tr, c].notna().mean() >= cfg.min_coverage]
        X = kb[feats]
        med, sd = X[tr].median(), X[tr].std().replace(0, 1)
        Z = ((X - med) / sd).fillna(0).clip(-5, 5)
        clf = LogisticRegression(C=0.05, max_iter=2000).fit(Z[tr], (y[tr] > 0).astype(int))
        q = (kb.index >= r0) & (kb.index < r1) & y.notna()
        if require_text:
            q &= kb["news_text"].str.len() > 0
        qi = np.flatnonzero(q)
        qi = qi[::step]
        if len(qi) == 0:
            continue
        p = clf.predict_proba(Z.iloc[qi])[:, 1]
        rows.append(pd.DataFrame(dict(p_up=p, exp_ret=p - 0.5, actual=y.iloc[qi].to_numpy(),
                                      base_rate=float((y[tr] > 0).mean())), index=kb.index[qi]))
    return pd.concat(rows)


def score(preds: pd.DataFrame) -> dict:
    a = preds["actual"].to_numpy()
    up = (a > 0).astype(int)
    p = preds["p_up"].to_numpy()
    base = preds["base_rate"].to_numpy()
    out = dict(n=len(preds),
               up_share=up.mean(),
               hit_rate=((p >= 0.5) == up).mean(),
               always_up_hit=up.mean(),
               auc=roc_auc_score(up, p) if 0 < up.sum() < len(up) else np.nan,
               ic=spearmanr(preds["exp_ret"], a).statistic,
               brier=np.mean((p - up) ** 2),
               brier_base=np.mean((base - up) ** 2))
    out["brier_skill"] = 1 - out["brier"] / out["brier_base"]
    return out


def strategy(close: pd.Series, preds: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Hold the index when P(up) >= threshold, else cash; re-decide at each query date."""
    r = close.pct_change().shift(-1)  # return earned from holding at close t to t+1
    # a decision is valid until the next query; if the model is silent for longer than
    # ~3 weeks (e.g. no headlines in a news-only window) default back to holding the index
    pos = (preds["p_up"] >= threshold).astype(float).reindex(close.index).ffill(limit=15).fillna(1.0)
    span = (close.index >= preds.index[0]) & (close.index <= preds.index[-1])
    r, pos = r[span].fillna(0), pos[span].fillna(0)
    return pd.DataFrame({"buy_hold": (1 + r).cumprod(), "rag_timing": (1 + r * pos).cumprod(),
                         "exposure": pos})


def perf(eq: pd.Series) -> dict:
    r = eq.pct_change().dropna()
    yrs = len(r) / 252
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, vol=r.std() * np.sqrt(252),
                sharpe=r.mean() / (r.std() + 1e-12) * np.sqrt(252), max_dd=(eq / eq.cummax() - 1).min())
