"""Synthetic-data tests: no look-ahead in retrieval, and the RAG recovers a planted signal."""

import numpy as np
import pandas as pd

from market_rag.features import TECH_COLUMNS, build_features
from market_rag.rag import MarketRAG, RAGConfig


def _synthetic(n=3000, seed=0, planted=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n)
    r = rng.normal(0.0003, 0.01, n)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=dates)
    news = []
    for i, d in enumerate(dates):
        # planted: "boom" headline days are followed by gains, "bust" by losses
        if planted and i % 40 == 0 and i + 25 < n:
            word = "boom rally" if (i // 40) % 2 == 0 else "bust crash"
            news.append((d, f"markets {word} today"))
            drift = 0.004 if "boom" in word else -0.004
            r[i + 1:i + 22] += drift
        news.append((d, f"filler headline {rng.integers(0, 50)} weather sports"))
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=dates)
    nd = pd.DataFrame(news, columns=["date", "headline"])
    return build_features(pd.DataFrame({"Close": close}), news=nd, news_window_days=1, horizons=(21,))


def test_no_lookahead():
    kb = _synthetic(planted=False)
    m = MarketRAG(RAGConfig(features=TECH_COLUMNS, use_news=False, tune=False)).fit(kb, kb.index[1500])
    for i in range(1500, 2500, 97):
        p = m.predict_index(i)
        assert (pd.to_datetime(kb.loc[p.neighbors.index, "label_date_21"]) < kb.index[i]).all()


def test_news_signal_recovered():
    kb = _synthetic()
    m = MarketRAG(RAGConfig(features=TECH_COLUMNS, alpha=1.0, k=5, tune=False, min_gap=1)).fit(kb, kb.index[2000])
    hits = []
    for i in range(2000, len(kb) - 25):
        txt = kb["news_text"].iloc[i]
        if "boom" in txt or "bust" in txt:
            p = m.predict_index(i)
            hits.append((p.exp_ret > 0) == ("boom" in txt))
    assert len(hits) > 10 and np.mean(hits) > 0.8
