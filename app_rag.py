"""Streamlit front-end for the news + metrics RAG.

    streamlit run app_rag.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_rag import data
from market_rag.backtest import perf, score, strategy, walk_forward
from market_rag.features import FEATURE_COLUMNS, NEWS_COLUMNS, build_features, describe_row, tone
from market_rag.rag import MarketRAG, RAGConfig

BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"

st.set_page_config(page_title="Market RAG", layout="wide")
st.title("News + Metrics RAG: what happened next in similar markets?")

st.sidebar.header("Knowledge base")
ticker = st.sidebar.text_input("Index / ticker", value="^GSPC").upper()
use_builtin_news = st.sidebar.checkbox("Built-in historical headlines (2008-2016, 2017-2020)", value=True)
st.sidebar.header("Retrieval")
horizon = st.sidebar.selectbox("Horizon (trading days)", [1, 7, 20, 63, 252], index=2)
use_news = st.sidebar.checkbox("Use news similarity", value=True)
tune = st.sidebar.checkbox("Learn alpha/k on validation slice", value=True)
k = st.sidebar.slider("k analogs (if not tuned)", 5, 60, 20)
alpha = st.sidebar.slider("alpha: news weight (if not tuned)", 0.0, 1.0, 0.25, 0.05)
what_if = st.sidebar.text_input("As-of date (blank = today/live)", value="")
st.sidebar.header("Backtest")
run_bt = st.sidebar.checkbox("Run walk-forward backtest", value=False)
bt_start = st.sidebar.number_input("Backtest start year", 1960, 2024, 2000)


@st.cache_data(show_spinner="Building knowledge base...")
def get_kb(ticker, builtin):
    px = data.load_prices(ticker)
    sh = data.load_shiller() if ticker in ("^GSPC", "SPY", "VOO", "^SPX") else None
    news = data.load_news(builtin=builtin)
    return build_features(px, sh, data.load_vix(), news, horizons=(1, 7, 20, 63, 252))


kb = get_kb(ticker, use_builtin_news)
cfg = RAGConfig(horizon=horizon, k=k, alpha=alpha if use_news else 0.0, tune=tune,
                use_news=use_news, min_coverage=0.3,
                features=FEATURE_COLUMNS if use_news else [c for c in FEATURE_COLUMNS if c not in NEWS_COLUMNS])

text = ""
if what_if:
    d = kb.index[kb.index.get_indexer([pd.Timestamp(what_if)], method="pad")[0]]
    model = MarketRAG(cfg).fit(kb, train_end=d)
    i = kb.index.get_loc(d)
    pred = model.predict_index(i)
    row, text = kb.iloc[i], kb["news_text"].iloc[i]
    realized = kb[f"fwd_ret_{horizon}"].iloc[i]
else:
    model = MarketRAG(cfg).fit(kb)
    row = kb.iloc[-1][FEATURE_COLUMNS].copy()
    try:
        live = data.fetch_live_news("SPY")
        text = " | ".join(live["headline"])
        if text:
            row["news_tone"], _ = tone(text)
            row["news_count"] = np.log1p(len(live))
    except Exception as e:
        st.info(f"Live headlines unavailable: {e}")
    pe = data.fetch_live_pe("SPY")
    if pe:
        row["pe_ttm"] = pe
    pred = model.predict_snapshot(row, text, as_of=kb.index[-1] + pd.Timedelta(days=1))
    pred.date, realized = kb.index[-1], np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"P(up) next {horizon}d", f"{pred.p_up:.0%}", f"{pred.p_up - model.base_rate:+.0%} vs base rate")
c2.metric("Weighted mean return", f"{pred.exp_ret:+.2%}")
c3.metric("Median analog return", f"{pred.median_ret:+.2%}")
c4.metric("Realized (what-if only)", "n/a" if pd.isna(realized) else f"{realized:+.2%}")
st.caption(model.summary())
st.markdown(f"**Snapshot {pred.date.date()}:** {describe_row(row)}")
if text:
    with st.expander("Headlines used for the query"):
        st.write(text[:3000])

# Fan of analog paths: what each analog did over the following horizon
close = kb["close"]
fig = go.Figure()
nb = pred.neighbors.sort_values("weight", ascending=False)
for d, r in nb.iterrows():
    j = close.index.get_loc(d)
    path = close.iloc[j:j + horizon + 1] / close.iloc[j] - 1
    color = BLUE if r[f"fwd_ret_{horizon}"] > 0 else ORANGE
    fig.add_trace(go.Scatter(x=np.arange(len(path)), y=path * 100, mode="lines",
                             line=dict(color=color, width=1 + 6 * r["weight"]), opacity=0.7,
                             name=str(d.date()), hovertemplate=f"{d.date()}<br>day %{{x}}: %{{y:.1f}}%"))
fig.add_hline(y=0, line_color=INK2, line_width=1)
fig.update_layout(title="What the retrieved analogs did next (blue = gain, orange = loss; width = weight)",
                  xaxis_title="trading days after analog date", yaxis_title="return %", showlegend=False,
                  height=420, margin=dict(t=50, b=40))
st.plotly_chart(fig, use_container_width=True)

show = nb[["weight", f"fwd_ret_{horizon}", "px_vs_ma50", "px_vs_ma200", "ret_1y", "pe_ttm", "cape", "vix",
           "news_sim"] + (["headlines"] if "headlines" in nb else [])].copy()
st.subheader("Retrieved analogs")
st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c != "headlines"}),
             use_container_width=True)

st.subheader("Learned metric weights")
st.bar_chart(model.weights.sort_values(ascending=False))

if run_bt:
    with st.spinner("Walk-forward backtest (refit yearly)..."):
        p = walk_forward(kb, cfg, f"{bt_start}-01-01", step=5, verbose=False)
    s = score(p)
    eq = strategy(close, p)
    st.subheader("Walk-forward backtest")
    a, b, c, dd = st.columns(4)
    a.metric("Hit rate", f"{s['hit_rate']:.1%}", f"always-up {s['always_up_hit']:.1%}", delta_color="off")
    b.metric("AUC", f"{s['auc']:.3f}")
    c.metric("Rank IC", f"{s['ic']:.3f}")
    dd.metric("Brier skill vs base rate", f"{s['brier_skill']:+.3f}")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=eq.index, y=eq["buy_hold"], name="buy & hold", line=dict(color=INK2, width=2)))
    f2.add_trace(go.Scatter(x=eq.index, y=eq["rag_timing"], name="RAG timing", line=dict(color=BLUE, width=2)))
    f2.update_layout(yaxis_type="log", title="Growth of $1 (hold when P(up) >= 0.5, else cash)", height=380,
                     hovermode="x unified")
    st.plotly_chart(f2, use_container_width=True)
    st.write({"RAG timing": perf(eq["rag_timing"]), "buy & hold": perf(eq["buy_hold"])})
