# StockCast

**StockCast** is an interactive stock price forecasting dashboard built with [Streamlit](https://streamlit.io/). It pulls historical prices for a public company, fits a lightweight additive time-series model (trend plus weekly and yearly seasonality), and charts the forecast with uncertainty intervals.

Type a ticker such as `AAPL` or a company name such as `apple`. StockCast resolves the symbol, fetches history from Yahoo Finance, and produces a forecast in seconds.

**GitHub:** [https://github.com/iqramushtaqk/stockcast](https://github.com/iqramushtaqk/stockcast)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)

---

## What It Does

1. Pulls historical closing prices from Yahoo Finance (with a public chart API fallback)
2. Fits an additive model: linear trend + weekly Fourier terms + yearly Fourier terms
3. Projects prices forward for a chosen number of days, with 95% residual intervals
4. Backtests on the last 30 days and reports RMSE
5. Renders interactive Plotly charts you can explore

Forecasts are for educational and portfolio use only — this is not financial advice.

## Features

- **Smart ticker resolution** — company name (`apple`, `tesla`) or ticker (`AAPL`, `TSLA`)
- **Configurable forecasts** — 7 to 365 days ahead
- **Adjustable history** — train on 1 to 10 years of past data
- **Confidence intervals** — likely range around the point forecast
- **Trend & seasonality** — what the model learned about long-term trend and yearly patterns
- **Backtested accuracy** — RMSE on a rolling 30-day holdout
- **CSV export** — dates, predicted price, upper/lower bounds

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Market data | yfinance, Yahoo Finance chart API fallback |
| Forecasting | NumPy least squares (additive seasonal model) |
| Charts | Plotly |

The model does **not** use Facebook Prophet or Stan, so Streamlit Community Cloud does not need a C++ compiler.

---

## How to Use

1. Open the app locally or on Streamlit Community Cloud
2. In the sidebar, type a ticker or company name
3. Set **Years of history** and **Days to forecast**
4. Optionally keep **Show trend & seasonality** on
5. Click **Run Forecast**
6. Download the forecast CSV if you want the numbers

---

## Run Locally

**Requirements:** Python 3.11

```bash
git clone https://github.com/iqramushtaqk/stockcast.git
cd stockcast

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

This repo is set up for a clean Cloud deploy (Python 3.11, no Prophet/cmdstan).

1. Push this repository to GitHub (already at [iqramushtaqk/stockcast](https://github.com/iqramushtaqk/stockcast))
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. **Create app** → select this repo, branch `main`, main file `app.py`
4. Deploy. Streamlit reads `runtime.txt` and `requirements.txt` automatically

No extra apt packages, secrets, or environment variables are required.

## Project Structure

```
stockcast/
├── app.py                  # UI, data fetch, model, charts
├── requirements.txt        # Python dependencies (Cloud-safe)
├── runtime.txt             # python-3.11 for Streamlit Cloud
├── .streamlit/config.toml  # dark theme, minimal toolbar
├── .gitignore
└── README.md
```

## Notes

- Fetched prices are cached for one hour.
- If `yfinance` is blocked, the app falls back to Yahoo’s public chart endpoint.
- Results depend on Yahoo Finance availability and are not guaranteed predictors of future prices.

---

Built as a compact Streamlit portfolio project: fetch → model → visualize → export in one `app.py`.
