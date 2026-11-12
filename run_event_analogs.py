"""Prototype B: connect news to similar historical moments.

Test 1 (events, 1980-2023): match each catalog event to earlier events by its card and/or the
         market's reaction; predict up/down over 1/7/20 days, the chance of a >5% / >10%
         drawdown within 20 days, and next-month volatility. Leave-future-out: an event only
         sees events whose 20-day outcome was known before it.
Test 2 (news days, held-out 2018-01..2020-06): does adding news themes and their trajectory
         (2-week slope) to the market metrics improve drawdown-risk and volatility forecasts?
         Train < 2015 <= validation (tuning) < 2018 <= test.

    python run_event_analogs.py      ->  reports/events_*.csv, reports/figs/events_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from market_rag import data
from market_rag.events import add_risk_labels, leave_future_out, load_events, score
from market_rag.features import TECH_COLUMNS, THEMES, VALUATION_COLUMNS, build_features
from market_rag.holdout import evaluate
from market_rag.rag import MarketRAG, RAGConfig

ROOT = Path(__file__).resolve().parent
REP, FIG = ROOT / "reports", ROOT / "reports" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
BLUE, VERM, GREEN, ORANGE, PINK, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#6b6b6b"


def load_kb():
    cache = data.CACHE_DIR / "kb.pkl"
    if cache.exists():
        kb = pd.read_pickle(cache)
        if "theme_pandemic_slope" in kb:
            return add_risk_labels(kb) if "fwd_dd_20" not in kb else kb
    px = data.load_prices("^GSPC")
    kb = build_features(px, data.load_shiller(), data.load_vix(), data.load_news(),
                        horizons=(1, 7, 20, 5, 21, 63, 252))
    add_risk_labels(kb)
    kb.to_pickle(cache)
    return kb


# ----------------------------------------------------------------------------- test 2
def risk_test(kb, val_start="2015-01-01", test_start="2018-01-01", test_end="2020-06-30"):
    METRICS = TECH_COLUMNS + VALUATION_COLUMNS
    themes = [f"theme_{t}" for t in THEMES] + [f"theme_{t}_nov" for t in THEMES]
    slopes = [f"theme_{t}_slope" for t in THEMES]
    news = kb["news_text"].str.len() > 0
    test = (kb.index >= test_start) & (kb.index <= test_end) & news.to_numpy() & kb["fwd_dd_20"].notna().to_numpy()
    qi = np.flatnonzero(test)
    feature_sets = {"market metrics": METRICS, "+ news themes": METRICS + themes,
                    "+ themes + trajectory": METRICS + themes + slopes}
    out_dd, out_vol = {}, {}
    for target in ("dd5", "vol"):
        k2 = kb.copy()
        # MarketRAG predicts the sign / mean of fwd_ret_20, so swap in the risk target:
        #   dd5: y > 0  <=>  no drawdown worse than -5% in the next 20 days
        #   vol: y = realised volatility over the next 20 days
        k2["fwd_ret_20"] = kb["fwd_dd_20"] + 0.05 if target == "dd5" else kb["fwd_vol_20"]
        for name, feats in feature_sets.items():
            cfg = RAGConfig(horizon=20, features=feats, use_news=False, min_coverage=0.3, k_grid=(10, 20, 40))
            m = MarketRAG(cfg).fit(k2, train_end=pd.Timestamp(test_start), val_start=pd.Timestamp(val_start))
            rows = []
            for i in qi:
                p = m.predict_index(i)
                rows.append((kb.index[i], 1 - p.p_up, p.exp_ret))
            df = pd.DataFrame(rows, columns=["date", "p", "e"]).set_index("date")
            if target == "dd5":
                out_dd[name] = df["p"]
            else:
                out_vol[name] = df["e"]
    # simple baselines
    tr = pd.to_datetime(kb["label_date_20"]) < pd.Timestamp(test_start)
    base = float((kb.loc[tr, "fwd_dd_20"] < -0.05).mean())
    from sklearn.linear_model import LogisticRegression

    X = np.log(kb[["vol_1m"]].clip(lower=0.02))
    ok = tr & X.notna().all(1) & kb["fwd_dd_20"].notna()
    clf = LogisticRegression().fit(X[ok], (kb.loc[ok, "fwd_dd_20"] < -0.05).astype(int))
    idx = kb.index[qi]
    out_dd["recent volatility only (logistic)"] = pd.Series(clf.predict_proba(X.loc[idx])[:, 1], index=idx)
    out_vol["recent volatility only"] = kb.loc[idx, "vol_1m"]
    y = kb.loc[idx, "fwd_dd_20"]
    rows = []
    for name, p in out_dd.items():
        r = evaluate(pd.DataFrame({"p_up": p, "actual": np.where(y < -0.05, 1.0, -1.0)}, index=idx), base)
        rows.append({"model": name, "target": "P(drawdown > 5% within 20d)", "n": r["n"], "event_rate": r["up_share"],
                     "auc": r["auc"], "auc_lo": r["auc_lo"], "auc_hi": r["auc_hi"], "brier_skill": r["brier_skill"]})
    for name, v in out_vol.items():
        rows.append({"model": name, "target": "next-20d volatility (rank corr.)", "n": len(v),
                     "auc": spearmanr(v, kb.loc[idx, "fwd_vol_20"]).statistic})
    return pd.DataFrame(rows), out_dd, idx


# ----------------------------------------------------------------------------- figures
def fig_events(preds, analogs, sc):
    # (a) Brier vs unconditional, (b) volatility rank correlation
    meths = ["all_events", "market_days", "shock_events", "card_events", "card_market"]
    labels = {"all_events": "any past event", "market_days": "similar market days",
              "shock_events": "events, similar market reaction", "card_events": "events, similar card",
              "card_market": "events, card + reaction"}
    cols = [GREY, BLUE, GREEN, VERM, ORANGE]
    tgts = [("p_up_1", "up\n1d"), ("p_up_7", "up\n7d"), ("p_up_20", "up\n20d"),
            ("p_dd5", "drop\n>5%"), ("p_dd10", "drop\n>10%")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(170 * mm, 80 * mm), layout="constrained",
                                 gridspec_kw={"width_ratios": [2.6, 1]})
    for j, (m, c) in enumerate(zip(meths, cols)):
        vals, lo, hi = [], [], []
        for t, _ in tgts:
            s = sc.loc[m, f"vs_uncond_{t}"]
            v, rest = s.split(" [")
            l, h = rest.rstrip("]").split(", ")
            vals.append(float(v)); lo.append(float(l)); hi.append(float(h))
        x = np.arange(len(tgts)) + (j - 2) * 0.15
        vals, lo, hi = map(np.array, (vals, lo, hi))
        a1.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="o", ms=4, color=c, lw=1, label=labels[m])
    a1.axhline(0, color=GREY, lw=0.8, ls="--")
    a1.set_xticks(range(len(tgts)), [t[1] for t in tgts], fontsize=8)
    a1.set_xlabel("target (drops = worst point within the next 20 trading days)", fontsize=8)
    a1.tick_params(axis="x", which="minor", bottom=False)
    a1.set_ylabel("Brier score minus base rate's\n(below 0 = better)")
    a1.set_title("76 events, 1980-2023, each predicted from earlier events only", fontsize=9.5, loc="left")
    a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    a1.spines[["top", "right"]].set_visible(False)
    v = [sc.loc[m, "vol_spearman"] for m in meths]
    a2.barh(range(len(meths)), v, color=cols)
    a2.set_yticks(range(len(meths)), [labels[m].replace(", ", ",\n") for m in meths], fontsize=7)
    a2.invert_yaxis()
    a2.axvline(0, color=GREY, lw=0.8)
    a2.set_xlabel("rank corr. with\nnext-month volatility")
    a2.tick_params(axis="y", which="minor", left=False)
    a2.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "events_scores.png"); fig.savefig(FIG / "events_scores.pdf"); plt.close(fig)

    # (c) reference classes for five events
    cases = ["2020-02-24", "2008-09-15", "2001-09-17", "2022-02-24", "2023-03-13"]
    fig, ax = plt.subplots(figsize=(170 * mm, 82 * mm), layout="constrained")
    for y0, d in enumerate(cases):
        d = pd.Timestamp(d)
        a = analogs[(d, "card_events")]
        ax.scatter(a["fwd_ret_20"] * 100, np.full(len(a), y0) + np.linspace(-0.18, 0.18, len(a)),
                   s=22, color=BLUE, alpha=0.8, label="matched past events" if y0 == 0 else None, zorder=3)
        ax.scatter(preds.loc[d, "fwd_ret_20"] * 100, y0, marker="D", s=46, color=VERM, zorder=4,
                   label="what actually happened" if y0 == 0 else None)
    short = {"2020-02-24": "COVID spreads to Italy/Korea", "2008-09-15": "Lehman bankruptcy",
             "2001-09-17": "Markets reopen after 9/11", "2022-02-24": "Russia invades Ukraine",
             "2023-03-13": "Silicon Valley Bank fails"}
    ax.set_yticks(range(len(cases)), [f"{short[d]}\n{d}" for d in cases], fontsize=7.5)
    ax.tick_params(axis="y", which="minor", left=False)
    ax.invert_yaxis()
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("index return over the next 20 trading days (%)")
    ax.set_title("Each event vs the 8 most similar earlier events by card", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "events_reference_classes.png"); fig.savefig(FIG / "events_reference_classes.pdf"); plt.close(fig)


def fig_by_card(ev):
    fig, axes = plt.subplots(1, 3, figsize=(170 * mm, 66 * mm), layout="constrained", sharey=True)
    for ax, (f, lab, names) in zip(axes, [
            ("trajectory", "trajectory on the day", {1: "emer-\nging", 2: "escal-\nating", 3: "climax", 4: "resol-\nving"}),
            ("severity", "perceived severity", {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}),
            ("tone", "tone", {-1: "bad", 0: "mixed", 1: "good"})]):
        g = ev.groupby(f).agg(n=("fwd_dd_20", "size"), dd5=("fwd_dd_20", lambda x: (x < -0.05).mean()))
        ax.bar(range(len(g)), g["dd5"] * 100, color=VERM, width=0.6)
        for i, (_, r) in enumerate(g.iterrows()):
            ax.text(i, r["dd5"] * 100 + 1.5, f"n={int(r['n'])}", ha="center", fontsize=7, color=GREY)
        ax.set_xticks(range(len(g)), [names[k] for k in g.index], fontsize=8)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_title(lab, fontsize=9.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    base = 14.6
    for ax in axes:
        ax.axhline(base, color=GREY, lw=0.8, ls="--")
    axes[2].text(2.45, base + 1.5, "all days", color=GREY, fontsize=7, ha="right")
    fig.suptitle("All 92 catalog events, in-sample. Cards were labelled in 2026, with hindsight.",
                 fontsize=9, x=0.01, ha="left")
    axes[0].set_ylabel("share of events followed by\na >5% drawdown within 20d (%)")
    fig.savefig(FIG / "events_by_card.png"); fig.savefig(FIG / "events_by_card.pdf"); plt.close(fig)


def fig_risk(res):
    d = res[res.target.str.startswith("P(")].set_index("model")
    fig, ax = plt.subplots(figsize=(150 * mm, 55 * mm), layout="constrained")
    cols = [BLUE, VERM, ORANGE, GREY]
    for j, (m, r) in enumerate(d.iterrows()):
        ax.errorbar(r["auc"], j, xerr=[[r["auc"] - r["auc_lo"]], [r["auc_hi"] - r["auc"]]], fmt="o",
                    color=cols[j % len(cols)], ms=5)
    ax.set_yticks(range(len(d)), d.index, fontsize=8)
    ax.tick_params(axis="y", which="minor", left=False)
    ax.invert_yaxis()
    ax.axvline(0.5, color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("AUC for a >5% drawdown within 20 days (95% CI)")
    ax.set_title("Held-out news days 2018-01..2020-06", fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "events_news_risk.png"); fig.savefig(FIG / "events_news_risk.pdf"); plt.close(fig)


def main():
    kb = load_kb()
    ev = load_events(kb)
    preds, analogs = leave_future_out(kb, ev)
    sc = score(preds)
    sc.to_csv(REP / "events_scores.csv")
    preds.to_csv(REP / "events_predictions.csv")
    pd.set_option("display.width", 220)
    print(f"Test 1: {len(preds)} events scored")
    print(sc[[c for c in sc.columns if c.startswith("brier")] + ["vol_spearman"]].round(3).to_string())
    res, _, _ = risk_test(kb)
    res.to_csv(REP / "events_news_risk.csv", index=False)
    print("\nTest 2: held-out news days")
    print(res.round(3).to_string(index=False))
    fig_events(preds, analogs, sc)
    fig_by_card(ev)
    fig_risk(res)
    print("figures:", sorted(p.name for p in FIG.glob("events_*.png")))


if __name__ == "__main__":
    main()
