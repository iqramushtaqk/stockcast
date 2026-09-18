# 📈 StockCast

**StockCast** is an interactive stock price forecasting dashboard built with [Streamlit](https://streamlit.io/) and [Facebook Prophet](https://facebook.github.io/prophet/). It pulls historical price data for any publicly traded company, fits a time-series forecasting model, and visualizes future price trends with confidence intervals — all in a clean, dark-themed interface.

Whether you type a ticker symbol like `AAPL` or just the company name like `apple`, StockCast automatically resolves it, fetches the data, and generates a forecast in seconds.

**🔗 Live demo:** *[link coming soon]*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)
![Prophet](https://img.shields.io/badge/Model-Facebook%20Prophet-green)

---

## What It Does

StockCast takes historical stock data and forecasts future prices using an additive time-series model. It's built to make forecasting approachable — no need to know exact ticker symbols, tune any parameters, or write a single line of code.

Under the hood, the app:

1. Pulls historical closing prices from Yahoo Finance
2. Fits a Prophet model to learn trend, weekly, and yearly seasonality
3. Projects prices forward for a chosen number of days
4. Backtests the model on the last 30 days to report accuracy (RMSE/MAE)
5. Renders everything as interactive charts you can explore

## Features

- 🔎 **Smart ticker resolution** — type a company name (`apple`, `tesla`) or a ticker (`AAPL`, `TSLA`) and it figures out the rest
- 🔮 **Configurable forecasts** — project anywhere from 7 to 365 days into the future
- 📅 **Adjustable history window** — train on 1 to 10 years of past data
- 📊 **Confidence intervals** — see the range of likely outcomes, not just a single prediction line
- 🧩 **Trend & seasonality breakdown** — visualize what the model learned about long-term trend and yearly patterns
- ✅ **Backtested accuracy** — RMSE and MAE reported on a rolling 30-day holdout, so you know how reliable the forecast is
- ⬇️ **CSV export** — download the full forecast (dates, predicted price, upper/lower bounds) for your own analysis

## Tech Stack

| Layer | Tool |
|---|---|
| UI / App framework | [Streamlit](https://streamlit.io/) |
| Market data | [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance) |
| Forecasting model | [Facebook Prophet](https://facebook.github.io/prophet/) |
| Charts | [Plotly](https://plotly.com/python/) |

---

## How to Use

1. Open the app (locally or via the live demo link above)
2. In the sidebar, type a **ticker symbol or company name** (e.g. `AAPL`, `apple`, `tesla`)
3. Adjust the sliders:
   - **Years of history** — how much past data to train on
   - **Days to forecast** — how far into the future to project
4. Toggle **Show trend & seasonality** if you want the breakdown charts
5. Click **🚀 Run Forecast**
6. Explore the results:
   - Metric cards for last close price, forecasted price, projected % change, and backtest accuracy
   - An interactive chart showing historical prices, the forecast, and the confidence interval
   - Optional trend and seasonality charts
7. Click **⬇️ Download forecast as CSV** to save the results

> **Note:** Forecasts are for educational and portfolio purposes only — this is not financial advice.

---

## Run Locally

**Requirements:** Python 3.10+

```bash
# clone the repo
git clone <your-repo-url>
cd stockcast

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Project Structure

```
stockcast/
├── app.py                  # entire app — UI, data fetch, model, charts
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Notes

- First run may take a few extra seconds while Prophet's backend compiles.
- The app caches fetched data for an hour to avoid redundant API calls.
- Model results depend on data availability from Yahoo Finance and are not guaranteed to be accurate predictors of future prices.

---

Built as a compact, single-file portfolio project — the entire pipeline (fetch → model → visualize → export) lives in one readable `app.py`.