"""Held-out evaluation of up/down over the next 1, 7 and 20 trading days.

    # metrics + news models, no LLM (free)
    python run_holdout.py

    # add the Claude learner (needs ANTHROPIC_API_KEY); responses are cached in data/cache/
    python run_holdout.py --llm --llm-max 300 --llm-val-max 80

    # check the LLM pipeline without an API key (stand-in model = analog vote)
    python run_holdout.py --llm --mock-llm

    # cleanest LLM test: a held-out period after the model's training cutoff (live yfinance data)
    python run_holdout.py --val-start 2023-01-01 --val-end 2026-07-01 --llm

Splits (defaults): train < 2015-01-01 <= validation < 2018-01-01 <= test <= end of data.
Writes reports/holdout_results.csv, reports/holdout_*.csv predictions and the LLM playbook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_rag import data
from market_rag.features import NEWS_COLUMNS, TECH_COLUMNS, VALUATION_COLUMNS, build_features
from market_rag.holdout import evaluate, llm_predict, logit_predict, rag_predict
from market_rag.llm import HORIZONS, MODEL, ClaudePredictor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
METRICS = TECH_COLUMNS + VALUATION_COLUMNS


def load_kb(source: str, rebuild: bool) -> pd.DataFrame:
    cache = data.CACHE_DIR / "kb.pkl"
    if cache.exists() and not rebuild:
        kb = pd.read_pickle(cache)
        if all(f"fwd_ret_{h}" in kb for h in HORIZONS) and "theme_pandemic_slope" in kb:
            return kb
    px = data.load_prices("^GSPC", source=source)
    kb = build_features(px, data.load_shiller(), data.load_vix(), data.load_news(),
                        horizons=(1, 7, 20, 5, 21, 63, 252))
    kb.to_pickle(cache)
    return kb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto", choices=["auto", "yfinance", "offline"])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--val-start", default="2015-01-01")
    ap.add_argument("--val-end", default="2018-01-01", help="= start of the held-out test")
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="low", help="per-forecast effort (playbook uses high)")
    ap.add_argument("--llm-step", type=int, default=3, help="query every N trading days")
    ap.add_argument("--llm-max", type=int, default=300, help="max held-out LLM queries")
    ap.add_argument("--llm-val-max", type=int, default=80, help="max validation queries for the playbook")
    ap.add_argument("--no-playbook", action="store_true")
    ap.add_argument("--headlines", action="store_true", help="show raw headlines to the LLM (leakage risk)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    kb = load_kb(args.source, args.rebuild)
    test_end = pd.Timestamp(args.test_end) if args.test_end else kb.index[-1]
    print(f"KB {kb.index[0].date()}..{kb.index[-1].date()} | train < {args.val_start} <= val < "
          f"{args.val_end} <= test <= {test_end.date()}")
    news_days = kb.index[kb["news_text"].str.len() > 0]

    results, preds_all = [], {}
    feats_news = METRICS + NEWS_COLUMNS
    for h in HORIZONS:
        tr = pd.to_datetime(kb[f"label_date_{h}"]) < pd.Timestamp(args.val_end)
        base = float((kb.loc[tr, f"fwd_ret_{h}"] > 0).mean())
        models = {
            "rag_metrics": lambda: rag_predict(kb, h, METRICS, args.val_start, args.val_end, test_end)[0],
            "rag_metrics_news": lambda: rag_predict(kb, h, feats_news, args.val_start, args.val_end,
                                                    test_end, use_news=True)[0],
            "logit_metrics_news": lambda: logit_predict(kb, h, feats_news, args.val_end, test_end),
        }
        for name, fn in models.items():
            print(f"[{h}d] {name} ...")
            p = fn()
            preds_all[(h, name)] = p
            p.to_csv(OUT / f"holdout_{name}_{h}d.csv")

    if args.llm:
        pred = ClaudePredictor(model=args.model, effort=args.effort, mock=args.mock_llm,
                               cache_path=data.CACHE_DIR / ("llm_cache_mock.jsonl" if args.mock_llm
                                                            else "llm_cache.jsonl"))
        tag = "llm_mock" if args.mock_llm else "llm"
        print(f"\n[{tag}] model={args.model if not args.mock_llm else 'MOCK'} effort={args.effort}")
        lp, playbook, recs, _, example = llm_predict(
            kb, METRICS + NEWS_COLUMNS, args.val_start, args.val_end, test_end, pred, step=args.llm_step,
            max_val=args.llm_val_max, max_test=args.llm_max, include_headlines=args.headlines,
            use_playbook=not args.no_playbook)
        (OUT / f"{tag}_playbook.md").write_text(playbook + "\n")
        (OUT / f"{tag}_prompt_example.txt").write_text(example + "\n")
        if recs is not None:
            recs.to_csv(OUT / f"{tag}_validation_records.csv")
        for h, p in lp.items():
            preds_all[(h, tag)] = p
            preds_all[(h, "analog_vote")] = p.assign(p_up=p["analog_vote"])
            p.to_csv(OUT / f"holdout_{tag}_{h}d.csv")
        if not args.mock_llm:
            u = pred.usage
            cost = u["input_tokens"] * 4e-6 + u["output_tokens"] * 20e-6
            print(f"  usage: {u} (~${cost:.2f} at Opus 5.5 list prices; cached calls are free)")

    llm_dates = {h: preds_all[(h, k)].index for (h, k) in preds_all if k.startswith("llm")}
    for (h, name), p in preds_all.items():
        tr = pd.to_datetime(kb[f"label_date_{h}"]) < pd.Timestamp(args.val_end)
        base = float((kb.loc[tr, f"fwd_ret_{h}"] > 0).mean())
        subsets = {"all_test": p, "news_days": p[p.index.isin(news_days)]}
        if h in llm_dates:
            subsets["llm_dates"] = p[p.index.isin(llm_dates[h])]
        for sname, sp in subsets.items():
            if len(sp) < 30:
                continue
            r = evaluate(sp, base)
            results.append({"horizon": f"{h}d", "model": name, "subset": sname, **r})

    res = pd.DataFrame(results)
    res.to_csv(OUT / "holdout_results.csv", index=False)
    show = res[["horizon", "subset", "model", "n", "always_up", "acc", "auc", "auc_lo", "auc_hi", "brier_skill"]]
    pd.set_option("display.width", 200)
    for s in show["subset"].unique():
        print(f"\n=== {s} ===")
        print(show[show.subset == s].drop(columns="subset").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
