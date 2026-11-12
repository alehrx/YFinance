"""Event analogs: match today's news *event* to similar historical events, not similar days.

Each catalog event (``events_catalog.csv``) has a hand-written card: type, economic channel,
scope, trajectory, perceived severity, policy support, tone and surprise. To that we attach
the market's reaction and starting point on the event day (computed from prices). An event
is then matched to earlier events on

    card      : what kind of event it is and how it hits the economy
    market    : how hard the market reacted and where it stood (drawdown, trend, volatility)

The output is a reference class: the outcomes that followed the matched events, i.e. up/down
over 1 / 7 / 20 trading days, the worst drawdown within 20 days, and realised volatility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATALOG = Path(__file__).resolve().parent / "events_catalog.csv"
TYPES = ["pandemic", "war", "terror", "financial", "monetary", "political", "trade", "energy",
         "sovereign", "market_structure", "disaster", "corporate"]
CHANNELS = ["supply", "demand", "financial_system", "rates", "confidence"]
MARKET = ["ret_1d", "ret_5d", "dd_52w", "px_vs_ma200", "vol_1m"]
OUTCOMES = ["fwd_ret_1", "fwd_ret_7", "fwd_ret_20", "fwd_dd_20", "fwd_vol_20"]


def add_risk_labels(kb: pd.DataFrame, h: int = 20) -> pd.DataFrame:
    """fwd_dd_{h}: worst close within the next h days vs today; fwd_vol_{h}: realised vol."""
    c = kb["close"]
    r = np.log(c).diff()
    kb[f"fwd_vol_{h}"] = r[::-1].rolling(h).std()[::-1].shift(-1) * np.sqrt(252)
    kb[f"fwd_dd_{h}"] = c[::-1].rolling(h).min()[::-1].shift(-1) / c - 1
    return kb


def load_events(kb: pd.DataFrame, path: Path = CATALOG) -> pd.DataFrame:
    ev = pd.read_csv(path, comment="#", parse_dates=["date"])
    ev = ev[(ev["date"] >= kb.index[0]) & (ev["date"] <= kb.index[-1])].copy()  # inside the data
    ev["mdate"] = kb.index[kb.index.searchsorted(ev["date"])]  # weekend / holiday -> next session
    if "fwd_dd_20" not in kb:
        add_risk_labels(kb)
    cols = MARKET + OUTCOMES + ["label_date_20", "vix", "close"]
    ev = ev.join(kb[cols], on="mdate")
    return ev.sort_values("mdate").reset_index(drop=True)


def card_matrix(ev: pd.DataFrame) -> np.ndarray:
    """Card -> vector. Type and channel overlaps dominate; ordinal fields add nuance."""
    t = np.array([[float(x in s.split(";")) for x in TYPES] for s in ev["types"]]) * 1.5
    c = np.array([[float(x in s.split(";")) for x in CHANNELS] for s in ev["channels"]])
    o = np.column_stack([
        (ev["scope"] - 1) / 3, (ev["trajectory"] - 1) / 3, (ev["severity"] - 1) / 4,
        ev["policy"] / 2, (ev["tone"] + 1) / 2, ev["surprise"]]) * 0.8
    return np.hstack([t, c, o])


def _market_matrix(ev: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    med = ref[MARKET].median()
    iqr = (ref[MARKET].quantile(0.75) - ref[MARKET].quantile(0.25)).replace(0, 1)
    return ((ev[MARKET] - med) / iqr).fillna(0).clip(-6, 6).to_numpy()


def _summ(out: pd.DataFrame, base: dict | None = None, m: float = 2.0) -> dict:
    """Outcome distribution of a set of analogs; probabilities shrunk toward ``base``."""
    n = len(out)
    res = {}
    for key, s in [("p_up_1", out["fwd_ret_1"] > 0), ("p_up_7", out["fwd_ret_7"] > 0),
                   ("p_up_20", out["fwd_ret_20"] > 0), ("p_dd5", out["fwd_dd_20"] < -0.05),
                   ("p_dd10", out["fwd_dd_20"] < -0.10)]:
        p = s.mean()
        res[key] = p if base is None else (n * p + m * base[key]) / (n + m)
    res["vol"] = out["fwd_vol_20"].median()
    res["n"] = n
    return res


def leave_future_out(kb: pd.DataFrame, ev: pd.DataFrame, k: int = 8, start: str = "1980-01-01",
                     market_days_k: int = 40) -> tuple[pd.DataFrame, dict]:
    """For every event from ``start`` on, predict its outcomes from strictly earlier information.

    Methods
      unconditional : all earlier trading days
      all_events    : all earlier events, no matching (does "something happened" matter?)
      market_days   : the 40 earlier *days* whose market state/reaction was most similar
      shock_events  : the k earlier *events* with the most similar market reaction (any type)
      card_events   : the k earlier events with the most similar card
      card_market   : card + market reaction, equally weighted
    """
    kbl = kb.dropna(subset=["fwd_ret_20", "fwd_dd_20", "fwd_vol_20"])
    kb_label = pd.to_datetime(kbl["label_date_20"]).to_numpy()
    C = card_matrix(ev)
    preds, analogs = [], {}
    for i, e in ev.iterrows():
        if e["mdate"] < pd.Timestamp(start) or pd.isna(e["fwd_ret_20"]):
            continue
        known = kb_label < np.datetime64(e["mdate"])
        past_days = kbl[known]
        base = _summ(past_days)
        prior = ev[(pd.to_datetime(ev["label_date_20"]) < e["mdate"])].index
        if len(prior) < 2 * k:
            continue
        # market-state matrices scaled on information available at the time
        Md = _market_matrix(past_days, past_days)
        md = _market_matrix(ev.loc[[i]], past_days)[0]
        dd = np.sqrt(((Md - md) ** 2).mean(1))
        nn_days = past_days.iloc[np.argsort(dd)[:market_days_k]]
        Me = _market_matrix(ev.loc[prior], past_days)
        dm = np.sqrt(((Me - md) ** 2).mean(1))
        dc = np.sqrt(((C[prior] - C[i]) ** 2).sum(1))
        z = lambda x: (x - x.mean()) / (x.std() + 1e-9)  # noqa: E731
        dcm = z(dc) + z(dm)
        row = {"date": e["mdate"], "name": e["name"]}
        for meth, sel in [("unconditional", None), ("all_events", ev.loc[prior]), ("market_days", nn_days),
                          ("shock_events", ev.loc[prior[np.argsort(dm)[:k]]]),
                          ("card_events", ev.loc[prior[np.argsort(dc)[:k]]]),
                          ("card_market", ev.loc[prior[np.argsort(dcm)[:k]]])]:
            r = base if sel is None else _summ(sel, base)
            for key, v in r.items():
                row[f"{meth}:{key}"] = v
            if meth in ("card_events", "card_market"):
                analogs[(e["mdate"], meth)] = sel[["mdate", "name"] + OUTCOMES]
        for o in OUTCOMES:
            row[o] = e[o]
        preds.append(row)
    return pd.DataFrame(preds).set_index("date"), analogs


def score(preds: pd.DataFrame, methods=("unconditional", "all_events", "market_days", "shock_events", "card_events",
                                         "card_market"), n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Brier score per target (lower is better), its bootstrap 95% CI of the difference
    vs unconditional, and the rank correlation of predicted vs realised volatility."""
    from scipy.stats import spearmanr

    targets = {"p_up_1": preds["fwd_ret_1"] > 0, "p_up_7": preds["fwd_ret_7"] > 0,
               "p_up_20": preds["fwd_ret_20"] > 0, "p_dd5": preds["fwd_dd_20"] < -0.05,
               "p_dd10": preds["fwd_dd_20"] < -0.10}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(preds), (n_boot, len(preds)))
    rows = []
    for m in methods:
        row = {"method": m}
        for t, y in targets.items():
            err = (preds[f"{m}:{t}"].to_numpy() - y.to_numpy().astype(float)) ** 2
            base = (preds[f"unconditional:{t}"].to_numpy() - y.to_numpy().astype(float)) ** 2
            row[f"brier_{t}"] = err.mean()
            d = (err - base)[idx].mean(1)
            row[f"vs_uncond_{t}"] = f"{(err - base).mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
        row["vol_spearman"] = spearmanr(preds[f"{m}:vol"], preds["fwd_vol_20"]).statistic
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")
