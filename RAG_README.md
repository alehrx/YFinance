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

RESULTS_PLACEHOLDER
