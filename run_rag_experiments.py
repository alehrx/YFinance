"""Reproduce the S&P 500 news+metrics RAG experiments.

    python run_rag_experiments.py            # yfinance prices, falls back to offline mirrors
    python run_rag_experiments.py --offline  # only the GitHub-hosted mirrors
    python run_rag_experiments.py --only news --news-horizons 5,21

Writes reports/rag_results.csv (+ per-experiment prediction CSVs and an equity-curve PNG).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_rag import data
from market_rag.backtest import logistic_baseline, perf, score, strategy, walk_forward
from market_rag.features import NEWS_COLUMNS, TECH_COLUMNS, VALUATION_COLUMNS, build_features
from market_rag.rag import RAGConfig

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
METRICS = TECH_COLUMNS + VALUATION_COLUMNS


def load_kb(offline: bool, rebuild: bool) -> pd.DataFrame:
    cache = data.CACHE_DIR / "kb.pkl"
    if cache.exists() and not rebuild:
        return pd.read_pickle(cache)
    px = data.load_prices("^GSPC", source="offline" if offline else "auto")
    kb = build_features(px, data.load_shiller(), data.load_vix(), data.load_news(),
                        horizons=(5, 21, 63, 252))
    kb.to_pickle(cache)
    return kb


def experiments(horizons_a, horizons_b):
    """(name, horizon, kind, cfg kwargs, walk-forward kwargs)."""
    exps = []
    for h in horizons_a:  # A: long history, metrics only. KB from 1951, out-of-sample 1990-2023.
        wf = dict(test_start="1990-01-01", step=5)
        exps += [
            (f"A{h}_rag_metrics", h, "rag", dict(features=METRICS, use_news=False), wf),
            (f"A{h}_rag_technicals", h, "rag", dict(features=TECH_COLUMNS, use_news=False), wf),
            (f"A{h}_logit", h, "logit", dict(features=METRICS), wf),
        ]
    for h in horizons_b:  # B: only on days that have dated headlines (2011-06..2016-06, 2018-06..2020-05).
        wf = dict(test_start="2011-06-01", test_end="2020-05-31", step=2, require_text=True)
        feats = METRICS + NEWS_COLUMNS
        exps += [
            (f"B{h}_rag_metrics", h, "rag", dict(features=METRICS, use_news=False), wf),
            (f"B{h}_rag_news_only", h, "rag", dict(features=feats, alpha_grid=(1.0,)), wf),
            (f"B{h}_rag_hybrid", h, "rag", dict(features=feats, alpha_grid=(0.0, 0.25, 0.5, 0.75)), wf),
            (f"B{h}_logit", h, "logit", dict(features=feats), wf),
        ]
    return exps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--only", default="", help="'metrics' or 'news'")
    ap.add_argument("--metric-horizons", default="21,63,252")
    ap.add_argument("--news-horizons", default="5,21")
    ap.add_argument("--summarize", action="store_true",
                    help="skip running; rebuild the results table from reports/preds_*.csv")
    ap.add_argument("--names", default="", help="run only these experiment names (comma list)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    kb = load_kb(args.offline, args.rebuild)
    ha = [] if args.only == "news" else [int(x) for x in args.metric_horizons.split(",") if x]
    hb = [] if args.only == "metrics" else [int(x) for x in args.news_horizons.split(",") if x]

    rows, curves = {}, {}
    names = set(filter(None, args.names.split(",")))
    for name, h, kind, ckw, wkw in experiments(ha, hb):
        if names and name not in names:
            continue
        cfg = RAGConfig(horizon=h, min_coverage=0.3, **ckw)
        print(f"\n=== {name} ===")
        pred_path = OUT / f"preds_{name}.csv"
        if args.summarize:
            if not pred_path.exists():
                continue
            p = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        elif kind == "rag":
            p = walk_forward(kb, cfg, verbose=False, **wkw)
        else:
            p = logistic_baseline(kb, cfg, **wkw)
        p.to_csv(pred_path)
        s = score(p)
        eq = strategy(kb["close"], p)
        curves[name] = eq
        s.update({f"strat_{k}": v for k, v in perf(eq["rag_timing"]).items()})
        s.update({f"bh_{k}": v for k, v in perf(eq["buy_hold"]).items()})
        s["exposure"] = eq["exposure"].mean()
        if "alpha" in p:
            s["mean_alpha"] = p["alpha"].mean()
            s["mean_k"] = p["k"].mean()
        rows[name] = s
        print({k: round(float(v), 3) for k, v in s.items()})

    res = pd.DataFrame(rows).T
    path = OUT / f"rag_results{'_' + args.only if args.only else ''}.csv"
    res.to_csv(path)
    print("\n" + res.round(3).to_string())
    print(f"\nwrote {path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
        groups = sorted({n.split("_")[0] for n in curves}, key=lambda g: (g[0], int(g[1:])))
        fig, axes = plt.subplots(len(groups), 1, figsize=(10, 3.2 * len(groups)), squeeze=False)
        for ax, g in zip(axes[:, 0], groups):
            members = [n for n in curves if n.split("_")[0] == g]
            bh = curves[members[0]]["buy_hold"]
            ax.plot(bh.index, bh, color="#52514e", lw=1.5, label="buy & hold")
            for c, n in zip(series, members):
                eq = curves[n]
                ax.plot(eq.index, eq["rag_timing"], color=c, lw=1.2, label=n.split("_", 1)[1])
            ax.set_yscale("log")
            ax.set_title(f"{g}: growth of $1, hold when P(up) >= 0.5 else cash", loc="left", fontsize=10)
            ax.grid(axis="y", color="#e6e5e0", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / f"rag_equity{'_' + args.only if args.only else ''}.png", dpi=110)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
