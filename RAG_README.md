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

## Held-out test: up or down over the next 1, 7 and 20 trading days (`run_holdout.py`)

| split | period | what it is used for |
|---|---|---|
| train | labels known before 2015-01-01 | metric scaling and weights, logistic fit, retrieval memory |
| validation | 2015-01-01 .. 2017-12-31 | tune the RAG (news weight α, k); the LLM forecasts here, sees its errors and writes a playbook |
| **test** | 2018-01-01 .. 2023-06-29 | never used for any choice. Parameters and playbook are frozen; retrieval memory grows only with outcomes already known on each query date |

News features now include **event themes**: the share of the week's headlines about pandemic, banking/credit,
war, trade, central bank, recession and energy, plus each theme's *novelty* (change vs the prior 3
months). With these, the query for 24 Feb 2020 retrieves Apr 2009 (swine flu) and Oct 2014 (Ebola) as
its closest analogs. Both were followed by gains, so the RAG says P(up 20d) = 77%. The actual move was
−31%. History offered no precedent for a virus that crashed the market.

### Results without the LLM (daily queries over the whole test period)

AUC is shown with a 95% block-bootstrap confidence interval. An interval that contains 0.5 means no evidence of skill.

| horizon | model | n | always-up | accuracy | AUC [95% CI] | Brier skill vs base rate |
|---|---|---|---|---|---|---|
| 1d | RAG metrics | 1381 | 53.7% | 50.8% | 0.511 [0.49, 0.55] | −0.034 |
| 1d | RAG metrics + news | 1381 | 53.7% | 51.3% | 0.513 [0.48, 0.54] | −0.033 |
| 1d | logistic, metrics + news | 1381 | 53.7% | 53.7% | 0.515 [0.48, 0.54] | −0.007 |
| 7d | RAG metrics | 1375 | 61.2% | 50.4% | **0.426 [0.38, 0.48]** | −0.100 |
| 7d | RAG metrics + news | 1375 | 61.2% | 53.9% | 0.485 [0.43, 0.54] | −0.059 |
| 7d | logistic, metrics + news | 1375 | 61.2% | 59.9% | 0.499 [0.42, 0.57] | −0.078 |
| 20d | RAG metrics | 1362 | 65.5% | 58.7% | 0.492 [0.42, 0.56] | −0.073 |
| 20d | RAG metrics + news | 1362 | 65.5% | 61.4% | 0.530 [0.45, 0.61] | −0.028 |
| 20d | logistic, metrics + news | 1362 | 65.5% | 64.8% | 0.538 [0.44, 0.65] | −0.052 |

On the 645 test days that have headlines (2018-01..2020-07), RAG metrics + news is the best variant at
every horizon (AUC 0.527 / 0.524 / 0.515), but every confidence interval still contains 0.5. The one
interval that excludes 0.5 is *below* it: the 7-day metrics RAG is anti-predictive (AUC 0.43). In the
2018-2023 test, dips bounced back within a week more often than the 1951-2017 analogs implied.

Full table: `reports/holdout_results.csv`. Predictions: `reports/holdout_*_{1,7,20}d.csv`.

### The LLM learner (`--llm`)

For each query, Claude receives the anonymized metrics and theme profile, 15 retrieved analogs with their
known 1/7/20-day outcomes, the base rates, and the **playbook** it wrote itself after reviewing its
validation-period mistakes. It returns P(up) for each horizon as structured JSON. The scoring is
identical to the models above, on the same dates.
See `reports/llm_prompt_example.txt` for exactly what the model sees.

```bash
pip install anthropic && export ANTHROPIC_API_KEY=...
python run_holdout.py --llm                 # 80 validation + 300 test queries, roughly $5-10 at Opus 5.5 list prices
python run_holdout.py --llm --no-playbook   # ablation: does the learned playbook help?
python run_holdout.py --llm --mock-llm      # pipeline check without an API key
```

Every response is cached in `data/cache/llm_cache.jsonl`, so reruns are free and an interrupted run resumes.
The learned playbook is saved to `reports/llm_playbook.md`, and `python -m market_rag.cli query --llm` then uses it for today's call.
Requests use server-side refusal fallback (`fallbacks="default"`).

**Leakage warning.** Claude knows market history up to its training cutoff. The prompts therefore carry no
dates, no index level and no raw headlines (unless you pass `--headlines`). Still, a profile like
"pandemic 43% of headlines, VIX 75" is recognizable. For a test the model cannot have memorized, use a
held-out period after its training cutoff, with live yfinance prices:

```bash
python run_holdout.py --source yfinance --val-start 2023-01-01 --val-end 2026-07-01 --llm
```

That forward period has no historical headline data (Yahoo only serves the latest ~20 items), so it
tests metrics plus analogs. To include news, collect headlines daily from now on (`--news my_news/*.csv`).

*Note:* sections A/B above were produced before the theme features were added and before a fix that
let the numeric news features (tone, themes) count in the distance. In those runs only the text
embedding carried news.

## Prototype B: match news *events* to historical events (`run_event_analogs.py`)

The idea is to retrieve similar historical **events**, not similar days.

- **Event catalog:** `market_rag/events_catalog.csv` lists 93 market-moving events from 1950 to 2023, from the
  Korean War and the Cuban missile crisis to Lehman, COVID and SVB. Each has a hand-written card: type,
  economic channel (supply / demand / financial system / rates / confidence), scope, trajectory
  (emerging / escalating / climax / resolving), perceived severity, policy support, tone and surprise.
- **Hindsight:** the cards were written in 2026, so the choice of events and their severity scores
  are coloured by what happened next.
- **Matching:** each event is matched to earlier events by card, by the market's reaction that day
  (1-day and 5-day move, drawdown, trend, volatility), or both.
- **Theme trajectory:** the news themes now also carry a trajectory feature, the change in each
  theme's headline share over the last two weeks.

**Test 1: 76 events, 1980-2023, each predicted only from events whose outcome was already known.**
Brier scores, where lower is better and 0.25 is a coin flip:

| method | up 1d | up 7d | up 20d | drop >5% in 20d | drop >10% | rank corr. with next-month vol |
|---|---|---|---|---|---|---|
| base rate (all earlier days) | 0.241 | 0.237 | **0.213** | 0.200 | 0.126 | 0.00 |
| any earlier event, no matching | **0.227** | 0.245 | 0.222 | 0.201 | **0.119** | −0.05 |
| similar market days | 0.264 | 0.239 | 0.235 | **0.194** | 0.125 | **0.67** |
| events with similar market reaction | 0.257 | 0.272 | 0.252 | 0.215 | 0.129 | 0.59 |
| events with similar card | 0.245 | 0.243 | 0.226 | 0.210 | 0.128 | 0.06 |
| events, card + reaction | 0.240 | 0.254 | 0.244 | 0.216 | 0.133 | 0.34 |

No method beats the base rate in a way the bootstrap confidence intervals support
(`reports/events_scores.csv`).

- **The matching itself works.** COVID (24 Feb 2020) pulls up Ebola, the 1957 and 1968 flus, H1N1 and
  SARS. Lehman pulls up LTCM, Continental Illinois, Bear Stearns and the BNP freeze. Ukraine pulls up
  Kuwait, Suez, Yom Kippur and the oil embargo.
- **Outcomes vary within each kind of event.** Lehman's eight matches had 20-day returns between
  −4% and +5%; Lehman itself fell 16%. COVID's ranged from −5% to +9%; COVID fell 31%.
  See `reports/figs/events_reference_classes.png`.
- **Volatility is predictable, from the market's reaction, not from the card.** Rank correlation with
  next-month volatility is 0.67 for similar market days and 0.59 for similar market reactions, against
  0.06 for card matching alone.

In-sample (`reports/figs/events_by_card.png`), events labelled *escalating* were followed by a
>5% drawdown 45% of the time, against 15% for all days. Severity 4-5 events were followed by one
57-80% of the time. Those severity scores were assigned with hindsight, and the pattern did not
survive the leave-future-out test.

**Test 2: held-out news days, 2018-01 to 2020-06 (train < 2015, tuning 2015-2017).** Can news
themes and their trajectory improve drawdown and volatility forecasts?

| features | AUC for a >5% drawdown within 20d [95% CI] | rank corr. with next-20d vol |
|---|---|---|
| market metrics | 0.51 [0.36, 0.69] | **0.52** |
| + news themes | 0.42 [0.29, 0.56] | 0.41 |
| + themes + trajectory | 0.42 [0.28, 0.56] | 0.42 |
| recent volatility only | 0.48 [0.30, 0.69] | 0.44 |

Adding news made both forecasts worse. Theme levels differ between the training headlines (Reddit
world news) and the test headlines (Reuters / CNBC / Guardian), and 628 overlapping days amount to
only about 30 independent months.

### Takeaways

1. "Find similar historical moments" works as **retrieval**: the matched events are sensible and
   make good context for a human or an LLM.
2. As a **forecast**, it does not beat the base rate on this data, for direction or for drawdowns.
   History mostly says "shocks recover", and the costly exceptions (1987, Lehman, COVID) are, almost
   by definition, the ones without close precedent.
3. The market's own reaction (size of the drop, volatility) is what predicts the *size* of the
   moves that follow. The news content adds little on top of that.
4. To test the news side properly would need consistent news coverage across many decades (for
   example NYT or WSJ archives) and event cards written blind to outcomes: an LLM reading only that
   day's headlines, validated after its training cutoff.

