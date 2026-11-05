"""Data loading: prices (yfinance or CSV), Shiller valuation data, VIX, and news.

Everything that is downloaded from the internet is cached under ``data/cache``
so repeated runs are fast and work offline.
"""

from __future__ import annotations

import csv
import glob
import io
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"

# Public datasets (GitHub-hosted, so reachable without API keys).
URLS = {
    # Shiller: monthly S&P 500 price, dividends, earnings, CPI, 10y rate, CAPE (1871-).
    "shiller": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    # CBOE VIX daily (1990-).
    "vix": "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv",
    # Offline fallback for S&P 500 daily prices when Yahoo is unreachable.
    "gspc_1950_2018": "https://media.githubusercontent.com/media/roman807/"
    "Predicting-Stock-Market-Crashes/master/data/%5EGSPC.csv",
    "gspc_2011_2023": "https://raw.githubusercontent.com/AnswerXuan/"
    "Galformer-Improved-Transformer-for-Time-Series-Prediction/main/Datasets/GSPC.csv",
    # Historical headline datasets (dated). Used to build the news side of the KB.
    "news_reuters": "https://raw.githubusercontent.com/JonPGallegos/SP500-Predictor-Using-NLP/"
    "main/data/reuters_headlines.csv",  # 2018-03 .. 2020-07
    "news_cnbc": "https://raw.githubusercontent.com/JonPGallegos/SP500-Predictor-Using-NLP/"
    "main/data/cnbc_headlines.csv",  # 2017-12 .. 2020-07
    "news_guardian": "https://raw.githubusercontent.com/JonPGallegos/SP500-Predictor-Using-NLP/"
    "main/data/guardian_headlines.csv",  # 2017-12 .. 2020-07
    "news_reddit": "https://media.githubusercontent.com/media/zhang-yiyao/FBDP/main/"
    "data/wc/RedditNews.csv",  # r/worldnews top-25 per day, 2008-06 .. 2016-07
}


def fetch(name: str, refresh: bool = False) -> Path:
    """Download dataset ``name`` from URLS into the cache (once) and return its path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.csv"
    if path.exists() and not refresh and path.stat().st_size > 1000:
        return path
    req = urllib.request.Request(URLS[name], headers={"User-Agent": "market-rag/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if data.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"{name}: got a Git LFS pointer instead of data")
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- prices
def _read_ohlc_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().title() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    out = df.set_index("Date")[[col]].rename(columns={col: "Close"})
    if "Volume" in df.columns:
        out["Volume"] = df.set_index("Date")["Volume"]
    return out.apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])


def load_prices(ticker: str = "^GSPC", source: str = "auto", csv_path: str | None = None,
                start: str = "1950-01-01") -> pd.DataFrame:
    """Daily Close (+Volume) indexed by date.

    source: "yfinance" | "csv" | "offline" (bundled GitHub mirrors, ^GSPC only) | "auto"
    ("auto" tries yfinance and falls back to the offline mirror for ^GSPC).
    """
    if csv_path:
        return _read_ohlc_csv(csv_path)
    if source in ("yfinance", "auto"):
        try:
            import yfinance as yf

            df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df[["Close", "Volume"]].dropna(subset=["Close"])
        except Exception as e:  # network blocked, rate limited, ...
            if source == "yfinance":
                raise
            print(f"[data] yfinance unavailable ({type(e).__name__}); using offline mirror")
        if source == "yfinance":
            raise RuntimeError(f"No data from yfinance for {ticker}")
    if ticker.upper() not in ("^GSPC", "SPX", "^SPX"):
        raise RuntimeError(f"Offline prices only bundled for ^GSPC, not {ticker}")
    a = _read_ohlc_csv(fetch("gspc_1950_2018"))
    b = _read_ohlc_csv(fetch("gspc_2011_2023"))
    px = pd.concat([a[a.index < b.index.min()], b]).sort_index()
    return px[~px.index.duplicated(keep="last")]


# --------------------------------------------------------------------------- valuation
def load_shiller() -> pd.DataFrame:
    """Monthly Shiller data: SP500 (monthly avg price), Earnings (TTM), PE10, Long Interest Rate."""
    df = pd.read_csv(fetch("shiller"), parse_dates=["Date"]).set_index("Date")
    df = df.replace(0.0, np.nan)  # trailing months are zero-filled in the source
    return df[["SP500", "Earnings", "PE10", "Long Interest Rate"]]


def load_vix() -> pd.Series:
    df = pd.read_csv(fetch("vix"), parse_dates=["DATE"]).set_index("DATE")
    return df["CLOSE"].rename("VIX").astype(float)


# --------------------------------------------------------------------------- news
_BYTES_PREFIX = re.compile(r"""^b(['"])(.*)\1$""", re.S)


def _clean_headline(h: str) -> str:
    h = str(h).strip()
    m = _BYTES_PREFIX.match(h)  # Reddit dump stores python bytes reprs: b'...'
    if m:
        h = m.group(2)
    h = h.replace("&amp;", "&").replace("\\'", "'").replace(" ,", ",").replace(" :", ":")
    return re.sub(r"\s+", " ", h)


def load_news_csv(path: str | Path, source: str | None = None) -> pd.DataFrame:
    """Load a two-column (date, headline) CSV, header optional. Returns date/headline/source."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(raw)))
    out = []
    for r in rows:
        if len(r) < 2:
            continue
        d = pd.to_datetime(r[0].strip(), errors="coerce")
        if pd.isna(d):
            continue  # header or junk line
        out.append((d.normalize(), _clean_headline(r[1])))
    df = pd.DataFrame(out, columns=["date", "headline"])
    df["source"] = source or Path(path).stem
    return df[df["headline"].str.len() > 5]


def load_news(paths: list[str] | None = None, builtin: bool = True) -> pd.DataFrame:
    """All headlines from user CSVs (glob patterns allowed) and/or the built-in datasets."""
    frames = []
    if builtin:
        for name in ("news_reddit", "news_reuters", "news_cnbc", "news_guardian"):
            try:
                frames.append(load_news_csv(fetch(name), source=name.replace("news_", "")))
            except Exception as e:
                print(f"[data] could not load {name}: {e}")
    for pattern in paths or []:
        for p in glob.glob(pattern):
            frames.append(load_news_csv(p))
    if not frames:
        return pd.DataFrame(columns=["date", "headline", "source"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "headline"])


def fetch_live_news(ticker: str = "SPY") -> pd.DataFrame:
    """Current headlines from Yahoo Finance (only ~10-20 recent items are exposed)."""
    import yfinance as yf

    items = yf.Ticker(ticker).news or []
    out = []
    for it in items:
        c = it.get("content", it)
        title = c.get("title")
        d = c.get("pubDate") or c.get("displayTime") or it.get("providerPublishTime")
        if title:
            d = pd.to_datetime(d, unit="s" if isinstance(d, (int, float)) else None, utc=True)
            out.append((d.tz_localize(None).normalize(), _clean_headline(title)))
    df = pd.DataFrame(out, columns=["date", "headline"])
    df["source"] = "yahoo"
    return df


def fetch_live_pe(ticker: str = "SPY") -> float | None:
    try:
        import yfinance as yf

        return yf.Ticker(ticker).info.get("trailingPE")
    except Exception:
        return None


def all_cached() -> list[str]:
    return sorted(os.listdir(CACHE_DIR)) if CACHE_DIR.exists() else []
