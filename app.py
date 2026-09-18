"""
StockCast — Stock Price Forecasting with Facebook Prophet
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StockCast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
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
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📈 StockCast")
st.sidebar.caption("Forecasting powered by Facebook Prophet")

ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").upper().strip()
years_back = st.sidebar.slider("Years of history", 1, 10, 5)
forecast_days = st.sidebar.slider("Days to forecast", 7, 365, 90)
show_components = st.sidebar.checkbox("Show trend & seasonality", value=True)

run = st.sidebar.button("🚀 Run Forecast", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:** historical prices are pulled from Yahoo Finance and "
    "fit with Prophet, an additive time-series model that captures trend, "
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

# Fast path for common companies — avoids a network round-trip for the usual suspects
COMMON_TICKERS = {
    "apple": "AAPL", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT", "tesla": "TSLA", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "disney": "DIS", "walmart": "WMT",
    "coca cola": "KO", "coca-cola": "KO", "nike": "NKE", "intel": "INTC",
    "ibm": "IBM", "boeing": "BA", "starbucks": "SBUX", "mcdonalds": "MCD",
    "mcdonald's": "MCD", "visa": "V", "mastercard": "MA", "paypal": "PYPL",
    "adobe": "ADBE", "salesforce": "CRM", "uber": "UBER", "airbnb": "ABNB",
    "spotify": "SPOT", "berkshire hathaway": "BRK-B",
}


def resolve_ticker(user_input: str) -> str:
    """Turn a company name or symbol the user typed into an actual ticker."""
    cleaned = user_input.strip()
    key = cleaned.lower()

    # 1) known company name
    if key in COMMON_TICKERS:
        return COMMON_TICKERS[key]

    # 2) looks like a real symbol already (short, no spaces) — try it as-is first
    if " " not in cleaned and len(cleaned) <= 6:
        test = yf.download(cleaned.upper(), period="5d", progress=False)
        if not test.empty:
            return cleaned.upper()

    # 3) fall back to Yahoo Finance's own search (handles names/typos)
    try:
        results = yf.Search(cleaned, max_results=1).quotes
        if results:
            return results[0]["symbol"]
    except Exception:
        pass

    # nothing matched — return the original input, upper-cased, and let the
    # normal "not enough data" error message handle it
    return cleaned.upper()


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(symbol: str, years: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        return df
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    return df


def fit_and_forecast(df: pd.DataFrame, periods: int):
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast


def backtest_rmse_mae(df: pd.DataFrame, holdout: int = 30):
    if len(df) < holdout + 30:
        return None, None
    train, test = df.iloc[:-holdout], df.iloc[-holdout:]
    model = Prophet(daily_seasonality=False)
    model.fit(train)
    future = model.make_future_dataframe(periods=holdout)
    fc = model.predict(future).tail(holdout)
    y_true = test["y"].values
    y_pred = fc["yhat"].values
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


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

    model, forecast = fit_and_forecast(data, forecast_days)
    rmse, mae = backtest_rmse_mae(data)

    last_price = data["y"].iloc[-1]
    predicted_price = forecast["yhat"].iloc[-1]
    pct_change = (predicted_price - last_price) / last_price * 100

    # --- metric cards -------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><h3>LAST CLOSE</h3><p>${last_price:,.2f}</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><h3>{forecast_days}-DAY FORECAST</h3><p>${predicted_price:,.2f}</p></div>""", unsafe_allow_html=True)
    with c3:
        color = "#22c55e" if pct_change >= 0 else "#ef4444"
        st.markdown(f"""<div class="metric-card"><h3>PROJECTED CHANGE</h3><p style="color:{color}">{pct_change:+.2f}%</p></div>""", unsafe_allow_html=True)
    with c4:
        rmse_txt = f"${rmse:,.2f}" if rmse else "n/a"
        st.markdown(f"""<div class="metric-card"><h3>BACKTEST RMSE (30d)</h3><p>{rmse_txt}</p></div>""", unsafe_allow_html=True)

    st.markdown("")

    # --- main forecast chart -------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["ds"], y=data["y"], name="Historical", line=dict(color="#8b5cf6", width=2)))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="#22c55e", width=2)))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(34,197,94,0.12)",
        line=dict(color="rgba(255,255,255,0)"), name="Confidence interval", showlegend=True,
    ))
    fig.update_layout(
        template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None, yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- components ------------------------------------------------------
    if show_components:
        st.subheader("Trend & seasonality")
        colA, colB = st.columns(2)
        with colA:
            trend_fig = go.Figure(go.Scatter(x=forecast["ds"], y=forecast["trend"], line=dict(color="#6366f1")))
            trend_fig.update_layout(template="plotly_dark", height=280, title="Trend",
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(trend_fig, use_container_width=True)
        with colB:
            if "yearly" in forecast.columns:
                season_fig = go.Figure(go.Scatter(x=forecast["ds"], y=forecast["yearly"], line=dict(color="#f59e0b")))
                season_fig.update_layout(template="plotly_dark", height=280, title="Yearly seasonality",
                                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(season_fig, use_container_width=True)

    # --- download ----------------------------------------------------
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