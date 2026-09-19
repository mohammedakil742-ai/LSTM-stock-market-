"""
===============================================================================
Stock Price Trend Prediction with LSTM — ALL IN ONE FILE
===============================================================================

Tools: Python, Keras, Pandas, Matplotlib, Yahoo Finance API (yfinance), Streamlit

Install:
    pip install yfinance pandas numpy matplotlib scikit-learn tensorflow streamlit

Run as CLI (fetch data, train, save weights + graphs):
    python stock_lstm_all_in_one.py --ticker AAPL

Run as Streamlit dashboard:
    streamlit run stock_lstm_all_in_one.py

Deliverables produced: model weights (AAPL_lstm.keras), graph (AAPL_prediction.png)
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["Close", "SMA20", "SMA50", "RSI14", "Volume"]


# ----------------------------------------------------------------------
# 1. Fetch data using yfinance API
# ----------------------------------------------------------------------
def fetch_data(ticker: str = "AAPL", start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and return a clean DataFrame."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}")
    # yfinance may return MultiIndex columns when a list of tickers is used
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ----------------------------------------------------------------------
# 2. Technical indicators: moving averages & RSI
# ----------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA20, SMA50, EMA20 and 14-period RSI (Wilder smoothing)."""
    out = df.copy()
    close = out["Close"]

    out["SMA20"] = close.rolling(20).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["RSI14"] = (100 - 100 / (1 + rs)).fillna(50.0)

    return out


# ----------------------------------------------------------------------
# 3. Normalize and prepare data (scalers fitted on train only — no leakage)
# ----------------------------------------------------------------------
def make_sequences(df: pd.DataFrame, look_back: int = 60, train_split: float = 0.8,
                   features: list[str] | None = None):
    """
    Returns X_train, y_train, X_test, y_test, feat_scaler, target_scaler, test_dates.
    """
    features = features or FEATURES
    data = df.dropna().copy()
    values = data[features].to_numpy(dtype="float32")
    target = data[["Close"]].to_numpy(dtype="float32")

    split = int(len(values) * train_split)
    if split <= look_back:
        raise ValueError("Not enough data for the chosen look_back / train_split.")

    feat_scaler = MinMaxScaler().fit(values[:split])
    target_scaler = MinMaxScaler().fit(target[:split])

    v = feat_scaler.transform(values)
    t = target_scaler.transform(target)

    X, y, dates = [], [], []
    for i in range(look_back, len(v)):
        X.append(v[i - look_back: i])
        y.append(t[i, 0])
        dates.append(data.index[i])

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="float32")
    dates = pd.DatetimeIndex(dates)

    cut = split - look_back  # index inside X where the test set starts
    return X[:cut], y[:cut], X[cut:], y[cut:], feat_scaler, target_scaler, dates[cut:]


# ----------------------------------------------------------------------
# 4. Build LSTM model using Keras
# ----------------------------------------------------------------------
def build_model(look_back: int, n_features: int):
    """Two stacked LSTM layers with dropout, single linear output."""
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            Input(shape=(look_back, n_features)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def train_model(model, X_train, y_train, epochs: int = 30, batch_size: int = 32,
                weights_path: str = "lstm_weights.keras"):
    """Train and validate the model; best weights saved to weights_path."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ----------------------------------------------------------------------
# 5. Evaluation & one-step forecast
# ----------------------------------------------------------------------
def evaluate(model, X_test, y_test, target_scaler):
    """Returns (actual_prices, predicted_prices, metrics_dict) in original units."""
    pred_scaled = model.predict(X_test, verbose=0)
    pred = target_scaler.inverse_transform(pred_scaled).ravel()
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    direction = float(np.mean(np.sign(np.diff(pred)) == np.sign(np.diff(actual))) * 100)
    return actual, pred, {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "DirAcc%": direction}


def forecast_next(model, X_test, target_scaler) -> float:
    """One-step-ahead prediction from the most recent window."""
    last = X_test[-1][np.newaxis, ...]
    return float(target_scaler.inverse_transform(model.predict(last, verbose=0))[0, 0])


# ----------------------------------------------------------------------
# 6a. CLI mode:  python stock_lstm_all_in_one.py --ticker AAPL
# ----------------------------------------------------------------------
def run_cli():
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Train an LSTM stock price model.")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--look-back", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    df = add_indicators(fetch_data(args.ticker, args.start))
    Xtr, ytr, Xte, yte, _fs, ts, dates = make_sequences(df, look_back=args.look_back)
    print(f"train={Xtr.shape}  test={Xte.shape}")

    model = build_model(args.look_back, Xtr.shape[2])
    model.summary()
    train_model(model, Xtr, ytr, epochs=args.epochs, weights_path=f"{args.ticker}_lstm.keras")

    actual, pred, metrics = evaluate(model, Xte, yte, ts)
    print(metrics)
    print("Next close prediction:", round(forecast_next(model, Xte, ts), 2))

    # Plot predictions vs actual
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, actual, label="Actual")
    ax.plot(dates, pred, label="Predicted")
    ax.set_title(f"{args.ticker} — LSTM predictions vs actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.ticker}_prediction.png", dpi=140)
    print(f"Saved {args.ticker}_prediction.png and {args.ticker}_lstm.keras")


# ----------------------------------------------------------------------
# 6b. Streamlit dashboard mode:  streamlit run stock_lstm_all_in_one.py
# ----------------------------------------------------------------------
def run_streamlit():
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="LSTM Stock Trend Predictor", layout="wide")
    st.title("Stock Price Trend Prediction with LSTM")

    with st.sidebar:
        ticker = st.text_input("Ticker", "AAPL").strip().upper()
        start = st.date_input("Start date", value=None, format="YYYY-MM-DD")
        look_back = st.slider("Look-back window (days)", 20, 120, 60, 10)
        epochs = st.slider("Epochs", 5, 60, 20, 5)
        run = st.button("Fetch & train", type="primary")

    @st.cache_data(show_spinner="Downloading prices...")
    def load(t: str, s: str):
        return add_indicators(fetch_data(t, s))

    @st.cache_resource(show_spinner="Training LSTM...")
    def train(t: str, s: str, lb: int, ep: int):
        df = load(t, s)
        Xtr, ytr, Xte, yte, _fs, ts, dates = make_sequences(df, look_back=lb)
        model = build_model(lb, Xtr.shape[2])
        train_model(model, Xtr, ytr, epochs=ep, weights_path=f"{t}_lstm.keras")
        actual, pred, metrics = evaluate(model, Xte, yte, ts)
        nxt = forecast_next(model, Xte, ts)
        return df, dates, actual, pred, metrics, nxt

    if not run:
        st.info("Choose a ticker in the sidebar and press **Fetch & train**.")
        return

    start_str = str(start) if start else "2015-01-01"
    try:
        df, dates, actual, pred, metrics, nxt = train(ticker, start_str, look_back, epochs)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build the model: {exc}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{metrics['RMSE']:.2f}")
    c2.metric("MAE", f"{metrics['MAE']:.2f}")
    c3.metric("MAPE", f"{metrics['MAPE%']:.2f}%")
    c4.metric("Next close (est.)", f"{nxt:.2f}")

    st.subheader("Predicted vs actual (test period)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, actual, label="Actual")
    ax.plot(dates, pred, label="Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Price with moving averages")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df.index, df["Close"], label="Close")
    ax2.plot(df.index, df["SMA20"], label="SMA 20")
    ax2.plot(df.index, df["SMA50"], label="SMA 50")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

    st.subheader("RSI (14)")
    fig3, ax3 = plt.subplots(figsize=(12, 2.5))
    ax3.plot(df.index, df["RSI14"], color="purple")
    ax3.axhline(70, ls="--", color="red")
    ax3.axhline(30, ls="--", color="green")
    ax3.set_ylim(0, 100)
    st.pyplot(fig3, clear_figure=True)

    st.caption("Educational project only — not investment advice.")


# ----------------------------------------------------------------------
# Entry point: Streamlit sets its own context; otherwise run the CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner_utils.script_run_context as _src
        in_streamlit = _src.get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False

    if in_streamlit:
        run_streamlit()
    else:
        run_cli()