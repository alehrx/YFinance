"""Figures for the held-out 1/7/20-day test (run_holdout.py must have been run first).

    python plot_holdout.py   ->  reports/figs/holdout_*.png / .pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
plt.style.use(str(ROOT / "journal.mplstyle"))

REP = ROOT / "reports"
FIG = REP / "figs"
FIG.mkdir(parents=True, exist_ok=True)
mm = 1 / 25.4
BLUE, VERM, GREEN, ORANGE, PINK, SKY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"
GREY = "#6b6b6b"
MODELS = {"rag_metrics": ("RAG metrics", BLUE), "rag_metrics_news": ("RAG metrics + news", VERM),
          "logit_metrics_news": ("Logistic, metrics + news", GREEN)}
HORIZONS = ["1d", "7d", "20d"]


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


res = pd.read_csv(REP / "holdout_results.csv")
kb = pd.read_pickle(ROOT / "data" / "cache" / "kb.pkl")

# 1) AUC with 95% CI ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(170 * mm, 72 * mm), layout="constrained", sharey=True)
for ax, (subset, title) in zip(axes, [("all_test", "All test days, 2018-01 to 2023-06"),
                                      ("news_days", "Test days with headlines, 2018-01 to 2020-07")]):
    for j, (m, (lab, col)) in enumerate(MODELS.items()):
        r = res[(res.subset == subset) & (res.model == m)].set_index("horizon").loc[HORIZONS]
        x = np.arange(3) + (j - 1) * 0.22
        ax.errorbar(x, r["auc"], yerr=[r["auc"] - r["auc_lo"], r["auc_hi"] - r["auc"]], fmt="o",
                    color=col, ms=5, lw=1.2, label=lab)
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--")
    ax.text(-0.45, 0.493, "no skill", color=GREY, fontsize=8, ha="left", va="top")
    ax.set_xticks(range(3), ["next 1 day", "next 7 days", "next 20 days"])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylim(0.22, 0.72)
    clean(ax)
axes[0].set_ylabel("AUC of P(up), 95% block-bootstrap CI")
axes[0].legend(loc="upper left", fontsize=8)
save(fig, "holdout_auc")

# 2) Accuracy vs always-up ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(170 * mm, 62 * mm), layout="constrained")
sub = res[res.subset == "all_test"]
for i, h in enumerate(HORIZONS):
    au = sub[sub.horizon == h]["always_up"].iloc[0]
    ax.hlines(au * 100, i - 0.38, i + 0.38, color=GREY, lw=2)
    if i == 0:
        ax.text(i - 0.38, au * 100 + 0.6, "always say 'up'", color=GREY, fontsize=8)
    for j, (m, (lab, col)) in enumerate(MODELS.items()):
        acc = sub[(sub.horizon == h) & (sub.model == m)]["acc"].iloc[0] * 100
        ax.bar(i + (j - 1) * 0.24, acc, width=0.2, color=col, label=lab if i == 0 else None)
ax.set_xticks(range(3), ["next 1 day", "next 7 days", "next 20 days"])
ax.tick_params(axis="x", which="minor", bottom=False)
ax.set_ylim(40, 70)
ax.set_ylabel("held-out accuracy (%)")
ax.legend(loc="upper left", ncol=3, fontsize=8)
clean(ax)
save(fig, "holdout_accuracy")

# 3) Timeline: index, model P(up 20d), and when the next 20 days were actually down ---
p20 = pd.read_csv(REP / "holdout_rag_metrics_news_20d.csv", index_col=0, parse_dates=True)
lg = pd.read_csv(REP / "holdout_logit_metrics_news_20d.csv", index_col=0, parse_dates=True)
close = kb["close"].loc[p20.index[0]:p20.index[-1]]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(170 * mm, 105 * mm), layout="constrained", sharex=True,
                             height_ratios=[1, 1.1])
down = p20["actual"] < 0
for ax in (a1, a2):
    # shade every test day whose next 20 trading days ended lower
    ax.fill_between(p20.index, 0, 1, where=down.to_numpy(), transform=ax.get_xaxis_transform(),
                    color=VERM, alpha=0.12, lw=0, step="mid")
a1.plot(close.index, close, color="black", lw=1)
a1.set_yscale("log")
a1.set_ylabel("S&P 500 (log)")
a1.set_title("Held-out test: shaded = the next 20 trading days ended lower", fontsize=10, loc="left")
a2.plot(p20.index, p20["p_up"].rolling(10).mean(), color=VERM, lw=1.2, label="RAG metrics + news")
a2.plot(lg.index, lg["p_up"].rolling(10).mean(), color=GREEN, lw=1.2, label="Logistic, metrics + news")
base = res[(res.horizon == "20d")]["always_up"].iloc[0]
a2.axhline(0.5, color=GREY, lw=0.8, ls="--")
a2.set_ylabel("P(up, next 20d)\n10-day average")
a2.set_ylim(0.2, 1.0)
a2.legend(loc="lower left", ncol=2, fontsize=8)
for d, lab, off in [("2018-12-24", "Dec 2018 sell-off", (-48, -6)), ("2020-03-23", "COVID crash", (42, 6)),
                    ("2022-01-03", "2022 bear market", (-62, -10))]:
    a1.annotate(lab, (pd.Timestamp(d), close.asof(pd.Timestamp(d))), xytext=off,
                textcoords="offset points", ha="center", va="center", fontsize=7.5, color=GREY,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=2, shrinkB=2))
clean(a1); clean(a2)
save(fig, "holdout_timeline")

# 4) Calibration at 20 days ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(85 * mm, 80 * mm), layout="constrained")
for m, (lab, col) in MODELS.items():
    p = pd.read_csv(REP / f"holdout_{m}_20d.csv", index_col=0)
    bins = pd.qcut(p["p_up"], 6, duplicates="drop")
    g = p.groupby(bins, observed=True).agg(pred=("p_up", "mean"), real=("actual", lambda a: (a > 0).mean()))
    ax.plot(g["pred"], g["real"], "o-", color=col, label=lab, ms=4, lw=1.2)
ax.plot([0.2, 1], [0.2, 1], color=GREY, lw=0.8, ls="--")
ax.text(0.98, 0.84, "perfect\ncalibration", color=GREY, fontsize=7.5, ha="right", va="top")
ax.set_xlim(0.2, 1); ax.set_ylim(0.2, 1)
ax.set_xlabel("predicted P(up, next 20d), sextile mean")
ax.set_ylabel("share that actually went up")
ax.legend(loc="upper left", fontsize=7.5)
clean(ax)
save(fig, "holdout_calibration_20d")

# 5) News themes vs the index, 2018-2020 ----------------------------------------------
w = kb.loc["2018-01-01":"2020-07-17"]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(170 * mm, 100 * mm), layout="constrained", sharex=True)
a1.plot(w.index, w["close"], color="black", lw=1)
a1.set_ylabel("S&P 500")
a1.set_title("What the news themes saw, 2018 to Jul 2020", fontsize=10, loc="left")
for t, col, lab in [("pandemic", VERM, "pandemic"), ("trade", BLUE, "trade / tariffs"),
                    ("central_bank", GREEN, "central bank / inflation"), ("recession", ORANGE, "recession / jobs")]:
    a2.plot(w.index, w[f"theme_{t}"] * 100, color=col, lw=1.2, label=lab)
a2.set_ylabel("share of the week's\nheadlines (%)")
a2.legend(loc="upper left", fontsize=8, ncol=2)
clean(a1); clean(a2)
save(fig, "news_themes_2018_2020")

# 6) The COVID query: what the analogs did vs what happened -----------------------------
from market_rag.features import FEATURE_COLUMNS  # noqa: E402
from market_rag.rag import MarketRAG, RAGConfig  # noqa: E402

q = pd.Timestamp("2020-02-24")
m = MarketRAG(RAGConfig(horizon=20, features=FEATURE_COLUMNS, min_coverage=0.3)).fit(kb, train_end=q)
pr = m.predict_index(kb.index.get_loc(q))
nb = pr.neighbors.sort_values("weight", ascending=False)
fig, ax = plt.subplots(figsize=(170 * mm, 78 * mm), layout="constrained")
allc = kb["close"]
for k, (d, r) in enumerate(nb.iterrows()):
    j = allc.index.get_loc(d)
    path = (allc.iloc[j:j + 21] / allc.iloc[j] - 1).to_numpy() * 100
    ax.plot(range(len(path)), path, color=BLUE, lw=0.6 + 5 * r["weight"], alpha=0.75)
    if k < 2:
        ax.annotate(f"{d:%b %Y}", (20, path[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=BLUE, va="center")
j = allc.index.get_loc(q)
act = (allc.iloc[j:j + 21] / allc.iloc[j] - 1).to_numpy() * 100
ax.plot(range(len(act)), act, color=VERM, lw=2.2)
ax.annotate(f"what actually happened\nafter 24 Feb 2020 ({act[-1]:+.0f}%)", (20, act[-1]), xytext=(-6, 6),
            textcoords="offset points", fontsize=8, color=VERM, va="bottom", ha="right")
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xlim(0, 26)
ax.set_xlabel("trading days after the query date")
ax.set_ylabel("index return (%)")
ax.set_title(f"24 Feb 2020: the {len(nb)} closest past days (blue, width = weight) said "
             f"P(up 20d) = {pr.p_up:.0%}", fontsize=10, loc="left")
clean(ax)
save(fig, "covid_analogs")
print("analogs:", [f"{d:%Y-%m-%d}" for d in nb.index[:5]], "p_up", round(pr.p_up, 2))
print("wrote", sorted(p.name for p in FIG.glob("*.png")))
