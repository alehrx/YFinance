"""Point-in-time features and forward-return labels.

Every feature on date t only uses information available at the close of t
(valuation data is lagged to when it would actually have been published).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TECH_COLUMNS = [
    "px_vs_ma50",     # close / 50-day SMA - 1
    "px_vs_ma200",    # close / 200-day SMA - 1
    "ma50_vs_ma200",  # golden / death cross distance
    "ret_1m",         # 21-day return
    "ret_3m",         # 63-day return
    "ret_1y",         # 252-day return ("price w.r.t. one year ago")
    "pos_52w",        # where in the 52-week range we are (0 = low, 1 = high)
    "dd_52w",         # drawdown from 52-week high
    "vol_1m",         # 21-day realised volatility, annualised
    "rsi14",
]
VALUATION_COLUMNS = [
    "pe_ttm",         # price / trailing-12m earnings
    "cape",           # Shiller cyclically-adjusted P/E
    "erp",            # earnings yield - 10y yield (crude equity risk premium)
    "rate_10y",
    "vix",
]
NEWS_COLUMNS = ["news_tone", "news_tone_chg", "news_count"]
FEATURE_COLUMNS = TECH_COLUMNS + VALUATION_COLUMNS + NEWS_COLUMNS

# Earnings for month m are reported ~1 quarter later; rates/CAPE price data ~1 month later.
EARNINGS_LAG_MONTHS = 3
MARKET_LAG_MONTHS = 1


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def technical_features(close: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=close.index)
    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    hi, lo = close.rolling(252).max(), close.rolling(252).min()
    f["px_vs_ma50"] = close / ma50 - 1
    f["px_vs_ma200"] = close / ma200 - 1
    f["ma50_vs_ma200"] = ma50 / ma200 - 1
    f["ret_1m"] = close.pct_change(21)
    f["ret_3m"] = close.pct_change(63)
    f["ret_1y"] = close.pct_change(252)
    f["pos_52w"] = (close - lo) / (hi - lo)
    f["dd_52w"] = close / hi - 1
    f["vol_1m"] = np.log(close).diff().rolling(21).std() * np.sqrt(252)
    f["rsi14"] = _rsi(close)
    return f


def valuation_features(close: pd.Series, shiller: pd.DataFrame | None,
                       vix: pd.Series | None, max_stale_months: int = 6) -> pd.DataFrame:
    f = pd.DataFrame(index=close.index, columns=VALUATION_COLUMNS, dtype=float)
    if shiller is not None:
        sh = shiller.copy()
        earn = sh["Earnings"].dropna()
        earn.index = earn.index + pd.DateOffset(months=EARNINGS_LAG_MONTHS)
        mkt = sh[["SP500", "PE10", "Long Interest Rate"]].dropna()
        mkt.index = mkt.index + pd.DateOffset(months=MARKET_LAG_MONTHS + 1)  # monthly avg known at month end
        limit = 22 * max_stale_months
        e = earn.reindex(close.index.union(earn.index)).ffill(limit=limit).reindex(close.index)
        m = mkt.reindex(close.index.union(mkt.index)).ffill(limit=limit).reindex(close.index)
        f["pe_ttm"] = close / e
        # CAPE scales with price; rescale the last published CAPE by price move since then.
        f["cape"] = m["PE10"] * close / m["SP500"]
        f["rate_10y"] = m["Long Interest Rate"]
        f["erp"] = 100 / f["pe_ttm"] - f["rate_10y"]
    if vix is not None:
        f["vix"] = vix.reindex(close.index).ffill(limit=5)
    return f


# A tiny finance-tone lexicon (subset in the spirit of Loughran-McDonald).
_POS = set("""rally rallies rallied surge surges surged soar soars soared gain gains gained jump jumps
jumped rise rises rising rebound rebounds rebounded record boom booming beat beats strong stronger
growth grow grows optimism optimistic upbeat recovery recover recovers bullish upgrade upgrades
profit profits deal deals agreement stimulus easing cut-rates hire hiring expansion boost boosts
boosted positive peace breakthrough approve approved approval vaccine highs win wins""".split())
_NEG = set("""fall falls fell plunge plunges plunged slump slumps slumped crash crashes crashed drop
drops dropped tumble tumbles tumbled sink sinks sank slide slides selloff sell-off loss losses lose
losing weak weaker recession fear fears worry worries worried crisis default downgrade downgrades
bearish slowdown slowing inflation layoffs layoff cuts bankruptcy bankrupt war attack attacks
sanctions tariff tariffs trade-war probe lawsuit fraud scandal panic volatile volatility turmoil
risk risks warning warns warned outbreak virus pandemic lockdown shutdown deficit lows protest
killed dead deaths collapse collapses""".split())
_TOKEN = re.compile(r"[a-z][a-z\-]+")


def tone(text: str) -> tuple[float, int]:
    toks = _TOKEN.findall(text.lower())
    p = sum(t in _POS for t in toks)
    n = sum(t in _NEG for t in toks)
    return (p - n) / (p + n + 1.0), p + n


def news_digest(news: pd.DataFrame, dates: pd.DatetimeIndex, window_days: int = 7,
                max_headlines: int = 200) -> pd.DataFrame:
    """Per trading day: concatenated headlines from the previous ``window_days`` calendar days
    (inclusive of the day itself) and simple tone statistics."""
    out = pd.DataFrame(index=dates, columns=["news_text", "news_tone", "news_count"])
    out["news_text"] = ""
    if news is None or news.empty:
        out[["news_tone", "news_count"]] = np.nan
        return out.assign(news_tone_chg=np.nan)
    by_day = news.groupby("date")["headline"].apply(list).sort_index()
    day_idx = by_day.index.values
    for d in dates:
        lo = np.searchsorted(day_idx, np.datetime64(d - pd.Timedelta(days=window_days - 1)), "left")
        hi = np.searchsorted(day_idx, np.datetime64(d), "right")
        if hi <= lo:
            continue
        heads = [h for lst in by_day.iloc[lo:hi][::-1] for h in lst][:max_headlines]
        txt = " | ".join(heads)
        out.at[d, "news_text"] = txt
        out.at[d, "news_tone"], _ = tone(txt)
        out.at[d, "news_count"] = len(heads)
    has = out["news_text"].str.len() > 0
    out.loc[~has, ["news_tone", "news_count"]] = np.nan
    out["news_tone"] = out["news_tone"].astype(float)
    out["news_count"] = np.log1p(out["news_count"].astype(float))
    out["news_tone_chg"] = out["news_tone"] - out["news_tone"].rolling(21, min_periods=5).mean()
    return out


def build_features(prices: pd.DataFrame, shiller: pd.DataFrame | None = None,
                   vix: pd.Series | None = None, news: pd.DataFrame | None = None,
                   horizons=(5, 21, 63), news_window_days: int = 7) -> pd.DataFrame:
    """Knowledge-base table: one row per trading day with features, news text and labels."""
    close = prices["Close"].astype(float)
    df = pd.concat([technical_features(close), valuation_features(close, shiller, vix)], axis=1)
    nd = news_digest(news, close.index, window_days=news_window_days)
    df = pd.concat([df, nd], axis=1)
    df["close"] = close
    for h in horizons:
        df[f"fwd_ret_{h}"] = close.shift(-h) / close - 1
        # date on which the label becomes known (used to prevent look-ahead in retrieval)
        df[f"label_date_{h}"] = pd.Series(close.index, index=close.index).shift(-h)
    df = df.iloc[252:]  # need a year of history for the 52w / 1y features
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    return df


def describe_row(row: pd.Series) -> str:
    """Compact human-readable snapshot, used in explanations and LLM prompts."""
    def pct(x):
        return "n/a" if pd.isna(x) else f"{x * 100:+.1f}%"

    def num(x, fmt="{:.1f}"):
        return "n/a" if pd.isna(x) else fmt.format(x)

    return (f"px/MA50 {pct(row['px_vs_ma50'])}, px/MA200 {pct(row['px_vs_ma200'])}, "
            f"MA50/MA200 {pct(row['ma50_vs_ma200'])}, 1m {pct(row['ret_1m'])}, "
            f"3m {pct(row['ret_3m'])}, 1y {pct(row['ret_1y'])}, 52w-pos {num(row['pos_52w'], '{:.2f}')}, "
            f"DD {pct(row['dd_52w'])}, vol {pct(row['vol_1m'])}, RSI {num(row['rsi14'], '{:.0f}')}, "
            f"P/E {num(row['pe_ttm'])}, CAPE {num(row['cape'])}, 10y {num(row['rate_10y'], '{:.2f}')}%, "
            f"VIX {num(row['vix'])}, news tone {num(row['news_tone'], '{:+.2f}')}")
