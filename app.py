"""
StockCast — Stock price forecasting with an additive time-series model.
Run: streamlit run app.py
"""

import json
import math
import warnings
from datetime import datetime, timedelta, timezone
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Page config (minimal toolbar hides Streamlit's Deploy button)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StockCast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1a1c24 0%, #23262f 100%);
        border: 1px solid #2d313d;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-card h3 { color: #8b949e; font-size: 13px; font-weight: 500; margin-bottom: 6px; }
    .metric-card p { color: #ffffff; font-size: 26px; font-weight: 700; margin: 0; }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📈 StockCast")
st.sidebar.caption("Forecasting with trend + weekly/yearly seasonality")

ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").upper().strip()
years_back = st.sidebar.slider("Years of history", 1, 10, 5)
forecast_days = st.sidebar.slider("Days to forecast", 7, 365, 90)
show_components = st.sidebar.checkbox("Show trend & seasonality", value=True)

run = st.sidebar.button("🚀 Run Forecast", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:** historical prices are pulled from Yahoo Finance and "
    "fit with an additive time-series model that captures trend, "
    "weekly/yearly seasonality, and produces uncertainty intervals."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Stock Price Forecasting")
st.caption("Type a ticker in the sidebar and hit **Run Forecast**.")

# ---------------------------------------------------------------------------
# Data + model helpers
# ---------------------------------------------------------------------------

COMMON_TICKERS = {
    "apple": "AAPL",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "disney": "DIS",
    "walmart": "WMT",
    "coca cola": "KO",
    "coca-cola": "KO",
    "nike": "NKE",
    "intel": "INTC",
    "ibm": "IBM",
    "boeing": "BA",
    "starbucks": "SBUX",
    "mcdonalds": "MCD",
    "mcdonald's": "MCD",
    "visa": "V",
    "mastercard": "MA",
    "paypal": "PYPL",
    "adobe": "ADBE",
    "salesforce": "CRM",
    "uber": "UBER",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "berkshire hathaway": "BRK-B",
}

YAHOO_UA = {"User-Agent": "Mozilla/5.0 (compatible; StockCast/1.0)"}
Z_95 = 1.96
WEEKLY_HARMONICS = 3
YEARLY_HARMONICS = 10


def _http_json(url: str, timeout: int = 20):
    req = Request(url, headers=YAHOO_UA)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _yahoo_chart_history(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    params = urlencode(
        {
            "period1": int(start.replace(tzinfo=timezone.utc).timestamp()),
            "period2": int(end.replace(tzinfo=timezone.utc).timestamp()),
            "interval": "1d",
            "events": "history",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?{params}"
    payload = _http_json(url)
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return pd.DataFrame()
    node = result[0]
    timestamps = node.get("timestamp") or []
    bars = ((node.get("indicators") or {}).get("quote") or [{}])[0]
    closes = bars.get("close") or []
    rows = [
        (datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None), close)
        for ts, close in zip(timestamps, closes)
        if close is not None and not (isinstance(close, float) and math.isnan(close))
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    return df


def _yahoo_search_symbol(query: str) -> str | None:
    url = (
        "https://query2.finance.yahoo.com/v1/finance/search?"
        + urlencode({"q": query, "quotesCount": 1, "newsCount": 0})
    )
    try:
        payload = _http_json(url)
        quotes = payload.get("quotes") or []
        if quotes and quotes[0].get("symbol"):
            return str(quotes[0]["symbol"]).upper()
    except (URLError, TimeoutError, ValueError, KeyError, OSError):
        return None
    return None


def _as_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _download_yfinance(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=_as_naive(start),
        end=_as_naive(end),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    close_col = "Close" if "Close" in df.columns else df.columns[-1]
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    out = df[[date_col, close_col]].copy()
    out.columns = ["ds", "y"]
    out["ds"] = pd.to_datetime(out["ds"]).dt.tz_localize(None)
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out.dropna()


def resolve_ticker(user_input: str) -> str:
    """Turn a company name or symbol the user typed into an actual ticker."""
    cleaned = user_input.strip()
    key = cleaned.lower()

    if key in COMMON_TICKERS:
        return COMMON_TICKERS[key]

    if " " not in cleaned and len(cleaned) <= 6:
        probe_end = datetime.now(timezone.utc)
        probe_start = probe_end - timedelta(days=10)
        try:
            test = _download_yfinance(cleaned.upper(), probe_start, probe_end)
            if not test.empty:
                return cleaned.upper()
        except Exception:
            pass
        try:
            test = _yahoo_chart_history(cleaned.upper(), probe_start, probe_end)
            if not test.empty:
                return cleaned.upper()
        except Exception:
            pass

    try:
        results = yf.Search(cleaned, max_results=1).quotes
        if results:
            return results[0]["symbol"]
    except Exception:
        pass

    found = _yahoo_search_symbol(cleaned)
    if found:
        return found

    return cleaned.upper()


def _design_matrix(dates: pd.Series, t0: pd.Timestamp) -> np.ndarray:
    days = (pd.to_datetime(dates) - t0).dt.total_seconds().to_numpy() / 86400.0
    cols = [np.ones(len(days)), days]
    for k in range(1, WEEKLY_HARMONICS + 1):
        cols.append(np.sin(2 * np.pi * k * days / 7.0))
        cols.append(np.cos(2 * np.pi * k * days / 7.0))
    for k in range(1, YEARLY_HARMONICS + 1):
        cols.append(np.sin(2 * np.pi * k * days / 365.25))
        cols.append(np.cos(2 * np.pi * k * days / 365.25))
    return np.column_stack(cols)


def _split_components(X: np.ndarray, beta: np.ndarray):
    intercept, slope = beta[0], beta[1]
    t = X[:, 1]
    trend = intercept + slope * t
    weekly_end = 2 + 2 * WEEKLY_HARMONICS
    weekly = X[:, 2:weekly_end] @ beta[2:weekly_end]
    yearly = X[:, weekly_end:] @ beta[weekly_end:]
    return trend, weekly, yearly


def fit_and_forecast(df: pd.DataFrame, periods: int):
    history = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    t0 = history["ds"].iloc[0]
    X = _design_matrix(history["ds"], t0)
    y = history["y"].to_numpy(dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    dof = max(len(y) - X.shape[1], 1)
    sigma = float(np.sqrt(np.sum(resid**2) / dof))

    last_date = history["ds"].iloc[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=periods, freq="D")
    all_dates = pd.concat([history["ds"], pd.Series(future_dates)], ignore_index=True)
    X_all = _design_matrix(all_dates, t0)
    yhat = X_all @ beta
    trend, weekly, yearly = _split_components(X_all, beta)
    band = Z_95 * sigma

    forecast = pd.DataFrame(
        {
            "ds": all_dates,
            "yhat": yhat,
            "yhat_lower": yhat - band,
            "yhat_upper": yhat + band,
            "trend": trend,
            "weekly": weekly,
            "yearly": yearly,
        }
    )
    return forecast, sigma


def backtest_rmse_mae(df: pd.DataFrame, holdout: int = 30):
    if len(df) < holdout + 30:
        return None, None
    train, test = df.iloc[:-holdout], df.iloc[-holdout:]
    fc, _ = fit_and_forecast(train, holdout)
    y_pred = fc.tail(holdout)["yhat"].to_numpy()
    y_true = test["y"].to_numpy()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(symbol: str, years: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)
    try:
        df = _download_yfinance(symbol, start, end)
        if not df.empty:
            return df
    except Exception:
        pass
    try:
        return _yahoo_chart_history(symbol, start, end)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if run:
    with st.spinner(f"Resolving '{ticker}' and fetching data..."):
        resolved = resolve_ticker(ticker)
        data = load_data(resolved, years_back)

    if resolved.upper() != ticker.upper():
        st.caption(f"Interpreted '{ticker}' as ticker **{resolved}**")
    ticker = resolved

    if data.empty or len(data) < 60:
        st.error(f"Couldn't find enough data for '{ticker}'. Check the symbol and try again.")
        st.stop()

    forecast, _sigma = fit_and_forecast(data, forecast_days)
    rmse, mae = backtest_rmse_mae(data)

    last_price = float(data["y"].iloc[-1])
    predicted_price = float(forecast["yhat"].iloc[-1])
    pct_change = (predicted_price - last_price) / last_price * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card"><h3>LAST CLOSE</h3><p>${last_price:,.2f}</p></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card"><h3>{forecast_days}-DAY FORECAST</h3><p>${predicted_price:,.2f}</p></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        color = "#22c55e" if pct_change >= 0 else "#ef4444"
        st.markdown(
            f"""<div class="metric-card"><h3>PROJECTED CHANGE</h3><p style="color:{color}">{pct_change:+.2f}%</p></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        rmse_txt = f"${rmse:,.2f}" if rmse else "n/a"
        st.markdown(
            f"""<div class="metric-card"><h3>BACKTEST RMSE (30d)</h3><p>{rmse_txt}</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["ds"],
            y=data["y"],
            name="Historical",
            line=dict(color="#8b5cf6", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Forecast",
            line=dict(color="#22c55e", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(34,197,94,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence interval",
            showlegend=True,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None,
        yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_components:
        st.subheader("Trend & seasonality")
        colA, colB = st.columns(2)
        with colA:
            trend_fig = go.Figure(
                go.Scatter(x=forecast["ds"], y=forecast["trend"], line=dict(color="#6366f1"))
            )
            trend_fig.update_layout(
                template="plotly_dark",
                height=280,
                title="Trend",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(trend_fig, use_container_width=True)
        with colB:
            season_fig = go.Figure(
                go.Scatter(x=forecast["ds"], y=forecast["yearly"], line=dict(color="#f59e0b"))
            )
            season_fig.update_layout(
                template="plotly_dark",
                height=280,
                title="Yearly seasonality",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(season_fig, use_container_width=True)

    st.markdown("")
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)
    out.columns = ["date", "predicted_price", "lower_bound", "upper_bound"]
    st.download_button(
        "⬇️ Download forecast as CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_forecast.csv",
        mime="text/csv",
    )

else:
    st.info("👈 Enter a ticker and click **Run Forecast** to get started. Try `AAPL`, `TSLA`, or `MSFT`.")
