# News + Metrics RAG for the S&P 500

This exercise asks one question: **when the market looked like it does today (in its metrics and
its headlines), what happened next?** It answers by retrieving similar days from the past
and looking at how they turned out.

## How it works

```
                 ┌───────────────────────── knowledge base (one doc per trading day) ─────────────────────────┐
 prices ───────▶ │ px/MA50, px/MA200, MA50/MA200, 1m/3m/1y return, 52w position, drawdown, vol, RSI          │
 Shiller data ─▶ │ trailing P/E, CAPE, earnings-yield − 10y, 10y yield  (lagged to publication date)          │
 VIX ──────────▶ │ VIX                                                                                         │
 headlines ────▶ │ digest of the previous 7 days of headlines → TF-IDF → LSA embedding, + tone score           │
                 │ label: forward return over h trading days (5 / 21 / 63 / 252), and the date it is known    │
                 └────────────────────────────────────────────────────────────────────────────────────────────┘
 query (today) ─▶ similarity = (1−α)·z(metric similarity) + α·z(news cosine)
                  candidates = only days whose outcome was already known on the query date
                  top-k analogs, de-duplicated so they come from different episodes (≥10 trading days apart)
                ─▶ P(up) = weighted share of analogs that gained (shrunk toward the base rate)
                   E[r]  = weighted mean analog return
                ─▶ optional: Claude reads the snapshot + analogs and writes a calibrated call (--llm)
```

What gets **learned** on each yearly refit, using only data from before the refit:

* robust scaling of every metric (median / IQR),
* how much each metric counts in the distance: its |rank correlation| with the forward return,
* the news embedding space (TF-IDF + truncated SVD fit on training headlines only),
* α (how much news similarity counts versus the metrics) and k, picked on a held-out validation
  slice at the end of the training window.

## Files

| Path | What it does |
|---|---|
| `market_rag/data.py` | loads prices (yfinance, or offline GitHub mirrors), Shiller P/E/CAPE, VIX, headline datasets, live Yahoo news |
| `market_rag/features.py` | point-in-time features, news digests, forward-return labels |
| `market_rag/rag.py` | `MarketRAG`: fit, retrieve, predict |
| `market_rag/backtest.py` | walk-forward evaluation, logistic-regression baseline, timing strategy |
| `market_rag/llm.py` | optional Claude generation step |
| `market_rag/cli.py` | `query` / `backtest` from the command line |
| `app_rag.py` | Streamlit UI (`streamlit run app_rag.py`) |
| `run_rag_experiments.py` | reproduces every number in the results below |
| `tests/test_market_rag.py` | checks that retrieval never looks ahead, and that a signal planted in synthetic news gets recovered |

## Usage

```bash
pip install -r requirements.txt
python -m market_rag.cli query                      # today: yfinance metrics + current Yahoo headlines
python -m market_rag.cli query --date 2020-03-23    # what-if on a past date (sees only what was known then)
python -m market_rag.cli query --horizon 63 --llm   # Claude makes the final call (needs `pip install anthropic` + API key)
python -m market_rag.cli backtest --horizon 21 --start 2000-01-01 --metrics-only
python run_rag_experiments.py                       # full experiment grid → reports/
streamlit run app_rag.py
```

To add your own headline history, pass CSVs of `date,headline` with `--news my_news/*.csv`.

## Results (S&P 500, walk-forward, refit every year, no look-ahead)

All numbers come from `python run_rag_experiments.py --offline`; the full table is in
`reports/rag_results.csv` and the growth-of-$1 curves are in `reports/rag_equity.png`.
"AUC" measures how well P(up) ranks winners above losers, where 0.5 means no skill. "Always-up"
is the hit rate of simply saying "up" every time. "Timing" means holding the index when
P(up) ≥ 0.5 and holding cash otherwise.

**A. Metrics only. Knowledge base from 1951, out-of-sample 1990-2023 (weekly queries)**

| horizon | model | hit rate | always-up | AUC | rank IC | timing Sharpe | buy & hold Sharpe | timing max DD | B&H max DD |
|---|---|---|---|---|---|---|---|---|---|
| 1 month | RAG, all metrics | 58.5% | 62.9% | 0.51 | +0.02 | 0.51 | 0.50 | −50% | −57% |
| 1 month | RAG, technicals only | 58.5% | 62.9% | 0.49 | −0.02 | 0.46 | 0.50 | −42% | −57% |
| 1 month | logistic regression | 58.1% | 62.9% | 0.48 | −0.03 | 0.38 | 0.50 | −56% | −57% |
| 3 months | RAG, all metrics | 62.6% | 69.1% | 0.46 | −0.02 | 0.38 | 0.49 | −56% | −57% |
| 3 months | RAG, technicals only | 64.0% | 69.1% | 0.52 | −0.01 | 0.46 | 0.49 | −49% | −57% |
| 1 year | RAG, all metrics | 61.6% | 78.9% | **0.26** | −0.26 | 0.43 | 0.49 | −57% | −57% |
| 1 year | RAG, technicals only | 73.5% | 78.9% | 0.46 | −0.02 | 0.48 | 0.49 | −57% | −57% |

**B. News + metrics. Only days that have dated headlines: 2011-06..2016-06 (Reddit world news) and 2018-06..2020-05 (Reuters/CNBC/Guardian)**

| horizon | model | hit rate | always-up | AUC | rank IC | learned α | timing Sharpe | B&H Sharpe |
|---|---|---|---|---|---|---|---|---|
| 1 week | RAG, metrics | 56.7% | 59.6% | **0.535** | **+0.08** | 0 | 0.63 | 0.62 |
| 1 week | RAG, news only | 54.7% | 59.6% | 0.49 | −0.05 | 1 | 0.51 | 0.62 |
| 1 week | RAG, hybrid | 55.4% | 59.6% | 0.53 | +0.08 | 0.25 | 0.56 | 0.62 |
| 1 week | logistic regression | 60.0% | 60.0% | 0.46 | −0.12 | – | 0.62 | 0.62 |
| 1 month | RAG, metrics | 62.2% | 65.3% | 0.50 | +0.01 | 0 | 0.52 | 0.62 |
| 1 month | RAG, news only | 59.8% | 65.3% | 0.46 | −0.15 | 1 | 0.58 | 0.62 |
| 1 month | RAG, hybrid | 60.8% | 65.3% | 0.46 | −0.07 | 0.25 | 0.48 | 0.62 |

### What the exercise shows

1. **No model beat "always up" on hit rate, and none showed real, repeatable skill.** All AUCs
   are within about ±0.04 of 0.5, and the labels overlap heavily (weekly queries of monthly
   returns), so the effective sample size is a few hundred, not 1,700. The best result, the
   1-week metrics RAG with AUC 0.535, is roughly a one-standard-error effect.
2. **The one consistent benefit is smaller drawdowns.** At 1 month the timing rule sat out about
   19% of the time, mostly in high-volatility, deep-drawdown stretches. That cut max drawdown
   from −57% to −42…−50% at about the same Sharpe. The model learned "volatility clusters",
   not "the market will fall".
3. **Retrieval cannot extrapolate.** The 1-year valuation model scored AUC 0.26, so it was
   reliably *wrong*: its average P(up) was 0.93 in 1999-2000, 0.89 in 2007 and 0.89 in 2021,
   and 0.42 in 2009 and 0.28 in 1992. Two things cause this. First, calm, trending markets had good next
   years in the knowledge base, so tops look bullish and crash bottoms look bearish. Second,
   CAPE in the 30s-40s had no precedent after 1951, so the "nearest" analogs came from another
   regime. A nearest-neighbour model is only as good as the history it can find.
4. **Headlines added nothing measurable** on these datasets. When given the choice, the tuner
   usually kept α at 0-0.25, and news-only retrieval was at or below chance. The caveats are
   real, though. The 2011-2016 headlines are general world news, not market news. The
   financial-news window is just two years. Bag-of-words embeddings capture topics
   ("virus", "tariff") rather than surprise relative to expectations, which is what moves
   prices.
5. **The machinery works when there is a signal to find.** `tests/test_market_rag.py` plants a
   headline that precedes moves in synthetic data, and the RAG recovers it more than 80% of the
   time. The null results above describe the market, not a bug.

### Ideas to push it further

* Better news data and embeddings: a modern sentence-embedding model over market
  headlines, earnings-call text or Fed statements, and scoring news *surprise* (today versus
  the trailing 30 days) instead of news content.
* Predict excess return versus the base rate, or risk (next-month volatility or drawdown),
  which item 2 suggests is the more learnable target.
* A cross-section of S&P 500 constituents (500 queries per day instead of 1) for far more
  independent samples.
* Use `--llm` on a sample of historical dates to measure whether an LLM reading the analogs is
  better calibrated than the weighted vote.

### Data caveats

Yahoo Finance was not reachable from the environment where these numbers were produced, so
prices came from GitHub mirrors of Yahoo's ^GSPC history (1950 to Jun 2023). Run it locally
and `load_prices` uses yfinance directly. P/E and CAPE come from Shiller's monthly data, with
earnings lagged 3 months to when they would actually have been published.

