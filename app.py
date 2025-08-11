# app.py — Stock News Sentiment: Spikes + Tiny Backtest (Phase 1)
# Features:
# - Sidebar: ticker, date range, headlines source (CSV or tiny sample)
# - Cleans headlines -> VADER sentiment -> daily average
# - Rolling z-score to flag sentiment spikes (pos/neg)
# - Tiny backtest: long after +spike, short after −spike (next day)
# - Metrics: hit rate, avg next-day after spikes, Sharpe (rough), Max Drawdown
# - Optional price-spike overlay
# - Pastel gradient background 🌈

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import date, timedelta, datetime
import re


# ---------------- UI Setup ----------------
st.set_page_config(page_title="Stock News Sentiment", page_icon="📈", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
  background: linear-gradient(120deg,#b3e5fc 0%,#f8bbd0 50%,#fff59d 100%);
}
.block-container{
  background: rgba(255,255,255,.65);
  backdrop-filter: blur(6px);
  border-radius: 16px;
  padding: 1.25rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock News Sentiment — Spikes & Backtest")

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start_dt, end=end_dt, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["Date", "Close"])
    df = df.rename_axis("Date").reset_index()[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df

@st.cache_resource(show_spinner=False)
def get_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()

def reshape_if_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts Kaggle Combined_News_DJIA.csv (Date + Top1..Top25) or already-long format.
       Returns columns: date, headline (string)."""
    cols = [c.lower() for c in df.columns]
    if "date" in cols and any(c.startswith("top") for c in cols):
        # Kaggle wide -> long
        df = df.rename(columns={c: c for c in df.columns})
        if "Date" in df.columns: df = df.rename(columns={"Date": "date"})
        head_cols = [c for c in df.columns if c.lower().startswith("top")]
        longdf = df.melt(id_vars=["date"], value_vars=head_cols, var_name="slot", value_name="headline")
        longdf = longdf.dropna(subset=["headline"])
        longdf["date"] = pd.to_datetime(longdf["date"], errors="coerce").dt.tz_localize(None)
        longdf["headline"] = longdf["headline"].astype(str)
        return longdf[["date", "headline"]].sort_values("date").reset_index(drop=True)
    # Assume already long with 'date','headline'
    out = df.rename(columns={df.columns[0]: "date", df.columns[1]: "headline"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out["headline"] = out["headline"].astype(str)
    return out.dropna(subset=["date", "headline"])

def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    px = df.copy()

    # If yfinance returned MultiIndex columns, flatten to top level
    if isinstance(px.columns, pd.MultiIndex):
        # keep first level name like 'Close', 'Open', etc.
        px.columns = [c[0] if isinstance(c, tuple) else c for c in px.columns]

    # Ensure there is a Date column (not index)
    if "Date" not in px.columns:
        # if Date is the index name or any index level, reset
        if px.index.name == "Date" or ("Date" in (px.index.names or [])):
            px = px.reset_index()
        else:
            px = px.reset_index()  # fallback: whatever the index is, make it a column named 'index'
            if "index" in px.columns:
                px = px.rename(columns={"index": "Date"})

    # Standardize names & types
    px = px.rename(columns={"Date": "date"})
    # Some installs return 'Close' or 'Adj Close' only — prefer Close if present
    close_cols = [c for c in px.columns if str(c).lower().startswith("close")]
    if not close_cols:
        raise ValueError("Could not find a Close column in price data.")
    px = px[["date", close_cols[0]]].rename(columns={close_cols[0]: "Close"})
    px["date"] = pd.to_datetime(px["date"], errors="coerce").dt.tz_localize(None)

    return px


def _mk_pattern(terms):
    terms = [t.strip() for t in (terms or []) if t and t.strip()]
    if not terms:
        return None
    # Escape special regex chars so terms are matched literally
    return "|".join(map(re.escape, terms))

def filter_headlines(df: pd.DataFrame, aliases: list[str], keep_keywords: list[str]) -> pd.DataFrame:
    patt_aliases = _mk_pattern(aliases)
    patt_keys    = _mk_pattern(keep_keywords)

    mask = pd.Series(True, index=df.index)
    if patt_aliases:
        mask &= df["headline"].str.contains(patt_aliases, case=False, na=False, regex=True)
    if patt_keys:
        mask &= df["headline"].str.contains(patt_keys, case=False, na=False, regex=True)

    return df.loc[mask].copy()


def daily_sentiment(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    tmp = df.copy()
    tmp["sentiment"] = tmp["headline"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    daily = tmp.groupby(tmp["date"].dt.date)["sentiment"].mean().rename("sentiment").to_frame()
    daily.index = pd.to_datetime(daily.index)
    # Align to business days and forward-fill gaps so returns align nicely
    daily = daily.asfreq("B").ffill()
    daily = daily.reset_index().rename(columns={"index": "date"})
    return daily  # columns: date, sentiment

def compute_z(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std()
    z = (series - mu) / sd.replace(0, np.nan)
    return z

def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = series / peak - 1
    return float(dd.min()) if len(series) else 0.0

# ---------------- Sidebar ----------------
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL")
today = date.today()
default_start = today - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

st.sidebar.header("Headlines Source")
mode = st.sidebar.radio("Mode", ["Historical CSV", "Tiny Sample"], index=0)

aliases_text = st.sidebar.text_input("Company aliases (comma-separated)", f"{ticker}, Apple, AAPL")
keyword_text = st.sidebar.text_input("Optional keywords to keep", "earnings, guidance, CEO, lawsuit, upgrade, downgrade, supply, strike")
aliases = [a.strip() for a in aliases_text.split(",") if a.strip()]
keep_keywords = [k.strip() for k in keyword_text.split(",") if k.strip()]

# ---------------- Load Prices ----------------
prices = load_prices(ticker, start_date, end_date)
if prices.empty:
    st.warning("No price data for this ticker/date range.")
    st.stop()

# ---------------- Load Headlines ----------------
headlines = pd.DataFrame(columns=["date", "headline"])
if mode == "Historical CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV ('date','headline') or Kaggle Combined_News_DJIA.csv", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded, encoding="latin-1")
        headlines = reshape_if_kaggle(raw)
elif mode == "Tiny Sample":
    sample_rows = [
        {"date": (datetime.now() - timedelta(days=9)).date(), "headline": f"{ticker} beats earnings; raises guidance"},
        {"date": (datetime.now() - timedelta(days=7)).date(), "headline": f"{ticker} supply chain disruption amid sector strike"},
        {"date": (datetime.now() - timedelta(days=5)).date(), "headline": f"{ticker} unveils new product; analysts upbeat"},
        {"date": (datetime.now() - timedelta(days=2)).date(), "headline": f"{ticker} faces lawsuit over antitrust concerns"},
        {"date": (datetime.now() - timedelta(days=1)).date(), "headline": f"{ticker} receives upgrade from major bank"},
    ]
    headlines = pd.DataFrame(sample_rows)
    headlines["date"] = pd.to_datetime(headlines["date"])

if headlines.empty:
    st.info("Load some headlines to continue (CSV or Tiny Sample).")
    st.stop()

# Filter to company & optional keywords
headlines = filter_headlines(headlines, aliases, keep_keywords)


# ---------------- Sentiment & Merge ----------------
analyzer = get_vader()
daily = daily_sentiment(headlines, analyzer)  # date, sentiment

px = normalize_prices(prices)
daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.tz_localize(None)



merged = (
    pd.merge(daily, px[["date", "Close"]], on="date", how="inner")
      .sort_values("date")
      .reset_index(drop=True)
)

if merged.empty:
    st.warning("No overlapping dates between sentiment and prices. Try a different date range or dataset.")
    st.stop()

# ---------------- Spike Detector ----------------
st.subheader("🔎 Sentiment Spikes")
colA, colB = st.columns(2)
with colA:
    win = st.slider("Rolling window (days)", 3, 20, 7)
with colB:
    thr = st.slider("Spike threshold |z|", 1.0, 4.0, 2.0, 0.1)

merged["z"] = compute_z(merged["sentiment"], win)
merged["pos_spike"] = merged["z"] > thr
merged["neg_spike"] = merged["z"] < -thr

spikes = merged.loc[merged["pos_spike"] | merged["neg_spike"], ["date", "sentiment", "z"]]
st.write("Detected spikes (recent):")
st.dataframe(spikes.tail(15), use_container_width=True)

# ---------------- Charts: Price & Sentiment ----------------
st.subheader("📊 Price & Sentiment")
chart_df = merged.set_index("date")[["Close", "sentiment"]]
st.line_chart(chart_df, height=280)

# ---------------- Tiny Backtest ----------------
st.subheader("🧪 Next-Day Spike Strategy vs Buy-&-Hold")

# Signal: long on +spike, short on −spike (trade next day)
signal_today = np.where(merged["pos_spike"], 1, np.where(merged["neg_spike"], -1, 0))
signal = pd.Series(signal_today, index=merged.index, name="signal")

# Next-day returns from Close-to-Close
ret_next = merged["Close"].pct_change().shift(-1).fillna(0.0)

strat_ret = (signal * ret_next).fillna(0.0)
bh_ret    = ret_next

perf = pd.DataFrame({
    "date": merged["date"],
    "Strategy": (1 + strat_ret).cumprod(),
    "Buy & Hold": (1 + bh_ret).cumprod()
}).dropna()

st.line_chart(perf.set_index("date"), height=280)

# ---------------- Metrics ----------------
hit_rate = float((np.sign(signal) * np.sign(ret_next) > 0).mean()) if len(signal) else 0.0
avg_next_pos = float(ret_next[merged["pos_spike"]].mean() * 100) if merged["pos_spike"].any() else 0.0
avg_next_neg = float(ret_next[merged["neg_spike"]].mean() * 100) if merged["neg_spike"].any() else 0.0
sharpe = float(np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-9)) if strat_ret.std() else 0.0
mdd = max_drawdown(perf["Strategy"]) if not perf.empty else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hit rate", f"{hit_rate:.0%}", help="Did signal predict next-day direction?")
c2.metric("Avg next-day after + spike", f"{avg_next_pos:.2f}%")
c3.metric("Avg next-day after − spike", f"{avg_next_neg:.2f}%")
c4.metric("Sharpe (rough)", f"{sharpe:.2f}")
st.caption(f"Max Drawdown (Strategy): {mdd:.1%}")

# ---------------- Optional: Price Spikes ----------------
with st.expander("➕ Compare to price spikes (optional)"):
    merged["ret"] = merged["Close"].pct_change()
    price_thr = st.slider("Price spike |z|", 1.5, 5.0, 2.5, 0.1, key="price_thr")
    merged["ret_z"] = compute_z(merged["ret"], win)
    merged["price_spike"] = merged["ret_z"].abs() > price_thr
    st.dataframe(
        merged.loc[(merged["pos_spike"] | merged["neg_spike"]) | merged["price_spike"],
                   ["date", "sentiment", "z", "ret", "ret_z", "price_spike"]].tail(20),
        use_container_width=True
    )


# ---------------- Notes ----------------
st.caption("Notes: This is exploratory. Headlines must be relevant to the ticker. Daily bars miss intraday reactions. Past performance ≠ future results.")
