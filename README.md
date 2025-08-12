# Stock Sentiment Spikes (Streamlit)

Detect unusual daily **news sentiment** (rolling z-score spikes) and test a tiny **next-day trading** rule vs **buy-and-hold**. Includes an **Event Study** (average abnormal returns around spikes) and an optional **FinBERT** model toggle.

> ⚠️ Educational project. Not investment advice.

---

## Features

- **Upload headlines** (CSV) or use a tiny built-in sample.
- **Sentiment scoring** with **VADER** (default). Optional **FinBERT** + ensemble weighting.
- **Spike detector:** rolling z-score on daily sentiment (configurable window & threshold).
- **Mini backtest:** long after positive spikes, short after negative spikes (next day).
- **Metrics:** hit rate, average next-day move after ± spikes, Sharpe (rough), max drawdown.
- **Event Study:** average cumulative abnormal return (CAR) from −k to +k business days vs a benchmark (e.g., `SPY`).
- **Optional** price-spike overlay for comparison.
  
---

## Quickstart

    # 1) Clone/download the repo, then:
    python -m venv .venv

    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    # source .venv/bin/activate

    pip install -r requirements.txt
    streamlit run app.py

The app launches with a tiny sample. For real runs, upload a CSV (see formats below).

---

## Data: headline options

### A) Kaggle DJIA headlines (easy, big)

**File:** `Combined_News_DJIA.csv`  
**Columns:** `Date`, `Label`, `Top1` … `Top25`

The app auto-reshapes this wide format into the required `date,headline` long format.

> Tip: Since these headlines are index-level (Dow Jones), set **Ticker** to `DIA` or `^DJI` for price alignment.  
> You can still filter by company aliases/keywords, but results may be noisier for single tickers.

### B) Your own CSV (simple)

Provide a CSV with exactly:

    date,headline
    2024-05-01,Apple beats on earnings and raises guidance
    2024-05-01,Analysts turn bullish on new iPhone cycle
    2024-05-02,Regulatory probe raises concerns

---

## How it works

1) **Score sentiment** per headline (VADER by default; optional FinBERT).  
2) **Aggregate** to daily average sentiment.  
3) Compute a **rolling z-score** → flag **positive/negative sentiment spikes** (days unusually high/low vs the last *k* days).  
4) **Backtest** a tiny rule:  
   - positive spike → **go long next day**  
   - negative spike → **go short next day**  
   Compare the strategy curve to buy-and-hold.  
5) **Event Study:** choose a benchmark (e.g., `SPY`) and plot the **average CAR** from −k to +k business days around spike dates.

---

## Usage

1) Pick **Ticker** and **Date range** in the sidebar.  
2) Choose **Headlines Source**:  
   - *Historical CSV* → upload `Combined_News_DJIA.csv` or your `date,headline` file  
   - *Tiny Sample* → quick local demo  
3) (Optional) Add **Company aliases** (e.g., `AAPL, Apple`) and **Keywords** (e.g., `earnings, guidance, lawsuit`) to filter headlines.  
4) Tune **Rolling window** and **Spike threshold (|z|)**.  
5) Review outputs:
   - **Detected spikes** table  
   - **Price & Sentiment** chart  
   - **Strategy vs Buy-&-Hold** chart + **Metrics**  
   - **Event Study** (CAR around spikes)  
6) (Optional) Toggle **Price spikes** to see if big sentiment days coincide with big return days.

---

## FinBERT (optional)

Turn on **Use FinBERT** in the sidebar to score with a finance-tuned model and blend with VADER.

### Install notes

Add these (already listed in `requirements.txt` if you followed the latest instructions):

    transformers>=4.42
    torch>=2.2
    accelerate>=0.30

If `torch` fails to install on Windows:

- Upgrade pip: `python -m pip install --upgrade pip`  
- Try CPU build:  
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`  
- Reduce **Max headlines to score** in the sidebar if inference is slow.

---

## Repo structure

    .
    ├─ app.py                # Streamlit app
    ├─ requirements.txt      # Streamlit, pandas, yfinance, VADER (+ optional FinBERT deps)
    ├─ .gitignore
    └─ README.md

---

## Screenshots 



---

## Parameters that matter

- **Rolling window (k):** how many past business days define “normal.”  
- **Spike threshold (|z|):** how extreme a day must be to count as a spike.  
- **FinBERT weight:** blend finance-tuned FinBERT with VADER (ensemble).  
- **Event window (±k):** days before/after spikes for the Event Study.  
- **Benchmark:** ticker for abnormal returns (default: `SPY`).

---

## Troubleshooting

- **No overlapping dates** → widen price date range or check CSV date formats/timezones.  
- **`MergeError / MultiIndex`** → ensure price columns are flattened and `date` is a plain column on both sides (the app includes a normalizer).  
- **`pandas has no attribute 'regex'`** → don’t use `pd.regex`; either set `regex=False` in `str.contains` or use `re.escape` with `regex=True`.  
- **Event Study `TypeError: 'str' object is not callable'`** → make sure `ar` is a **Series** (squeeze DataFrames before naming).  
- **FinBERT slow/failing** → leave it off, limit max rows, or install CPU-only torch.

---

## Roadmap

- Live news mode (NewsAPI/Finnhub/Polygon) with caching  
- Intraday prices + true **VWAP** overlay  
- Spearman vs Pearson & bootstrap CIs for lagged correlations  
- Granger causality test and richer event categories

---

## License

MIT 

---

## Acknowledgments

- **VADER:** Hutto & Gilbert (2014)  
- **FinBERT:** ProsusAI  
- **Price data:** `yfinance` (Yahoo Finance)
