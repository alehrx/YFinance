"""Command line: query today's (or any date's) analogs, or run a quick backtest.

    python -m market_rag.cli query                    # live: yfinance prices + Yahoo headlines
    python -m market_rag.cli query --date 2020-03-20  # historical what-if (uses only info before it)
    python -m market_rag.cli query --llm              # let Claude write the final call
    python -m market_rag.cli backtest --horizon 21 --start 2000-01-01
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import data
from .backtest import perf, score, strategy, walk_forward
from .features import FEATURE_COLUMNS, NEWS_COLUMNS, build_features, describe_row, news_digest, tone
from .rag import MarketRAG, RAGConfig


def build_kb(args) -> pd.DataFrame:
    px = data.load_prices(args.ticker, source=args.source, csv_path=args.prices_csv)
    shiller = data.load_shiller() if args.ticker.upper() in ("^GSPC", "SPY", "^SPX", "VOO") else None
    news = data.load_news(args.news, builtin=not args.no_builtin_news)
    return build_features(px, shiller, data.load_vix(), news, horizons=(5, 21, 63, 252))


def cmd_query(args):
    kb = build_kb(args)
    cfg = RAGConfig(horizon=args.horizon, k=args.k, alpha=args.alpha, tune=not args.no_tune,
                    min_coverage=0.3)
    if args.date:  # historical what-if: model and KB only see what was known on that date
        d = kb.index[kb.index.get_indexer([pd.Timestamp(args.date)], method="pad")[0]]
        model = MarketRAG(cfg).fit(kb, train_end=d)
        i = kb.index.get_loc(d)
        pred = model.predict_index(i)
        row, text = kb.iloc[i], kb["news_text"].iloc[i]
    else:  # live: latest metrics + current Yahoo headlines
        model = MarketRAG(cfg).fit(kb)
        row = kb.iloc[-1][FEATURE_COLUMNS].copy()
        try:
            live = data.fetch_live_news(args.news_ticker)
        except Exception as e:
            print(f"(no live news: {e})")
            live = pd.DataFrame(columns=["date", "headline"])
        text = " | ".join(live["headline"].tolist())
        if text:
            row["news_tone"], _ = tone(text)
            row["news_count"] = np.log1p(len(live))
        pe = data.fetch_live_pe(args.news_ticker)
        if pe and (pd.isna(row["pe_ttm"]) or args.ticker.upper() in ("^GSPC", "SPY")):
            row["pe_ttm"] = pe
        pred = model.predict_snapshot(row, text, as_of=kb.index[-1] + pd.Timedelta(days=1))
        pred.date = kb.index[-1]
    print(model.summary())
    print(f"\nQuery {pred.date.date()}: {describe_row(row)}")
    if text:
        print(f"Headlines: {text[:400]}...")
    print(f"\n==> {args.horizon}-day outlook: P(up) = {pred.p_up:.0%}  "
          f"(base rate {model.base_rate:.0%}), weighted mean {pred.exp_ret:+.2%}, "
          f"median {pred.median_ret:+.2%}  [alpha={pred.alpha}, k={pred.k}]")
    nb = pred.neighbors.sort_values("weight", ascending=False)
    cols = ["weight", f"fwd_ret_{args.horizon}", "px_vs_ma50", "px_vs_ma200", "ret_1y", "pe_ttm", "vix",
            "news_sim"]
    cols = [c for c in cols if c in nb]
    print("\nTop analogs:\n" + nb[cols].head(10).round(3).to_string())
    if args.llm:
        from .llm import build_prompt, llm_call

        prompt = build_prompt(row, text, pred, args.horizon, model.base_rate)
        out = llm_call(prompt)
        print(f"\nClaude: P(up)={out['p_up']:.0%}, E[r]={out['expected_return_pct']:+.2f}%, {out['stance']}")
        print("Drivers:", "; ".join(out["key_drivers"]))
        print(out["rationale"])


def cmd_backtest(args):
    kb = build_kb(args)
    cfg = RAGConfig(horizon=args.horizon, min_coverage=0.3, use_news=not args.metrics_only)
    if args.metrics_only:
        cfg.features = [c for c in FEATURE_COLUMNS if c not in NEWS_COLUMNS]
    p = walk_forward(kb, cfg, args.start, args.end, step=args.step,
                     require_text=args.require_text)
    s = score(p)
    eq = strategy(kb["close"], p)
    print({k: round(float(v), 3) for k, v in s.items()})
    print("timing:", {k: round(v, 3) for k, v in perf(eq["rag_timing"]).items()})
    print("buy&hold:", {k: round(v, 3) for k, v in perf(eq["buy_hold"]).items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["query", "backtest"])
    ap.add_argument("--ticker", default="^GSPC")
    ap.add_argument("--source", default="auto", choices=["auto", "yfinance", "offline"])
    ap.add_argument("--prices-csv")
    ap.add_argument("--news", nargs="*", help="extra headline CSVs (date,headline); globs ok")
    ap.add_argument("--no-builtin-news", action="store_true")
    ap.add_argument("--news-ticker", default="SPY", help="ticker whose Yahoo headlines/PE are used live")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--date")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--require-text", action="store_true")
    args = ap.parse_args()
    {"query": cmd_query, "backtest": cmd_backtest}[args.command](args)


if __name__ == "__main__":
    main()
