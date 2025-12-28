#!/usr/bin/env python3
"""
oil_price_forecast.py
CODE A – Clean, robust Brent/WTI forecast

- Data: Yahoo Finance (WTI + Brent)
- Features: returns, lags, Brent–WTI spread
- Simple RandomForest
- Trend filter
- TXT output (overwritten every run)
"""

import json
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# ======================
# CONFIG
# ======================
WTI_SYMBOL = "CL=F"
BRENT_SYMBOL = "BZ=F"
START_DATE = "2015-01-01"

OUTPUT_TXT = "oil_forecast_output.txt"

PROB_TRADE_THRESHOLD = 0.57
MIN_ROWS = 250

# ======================
# DATA LOADING
# ======================
def load_prices():
    wti = yf.download(WTI_SYMBOL, start=START_DATE, progress=False, auto_adjust=True)
    brent = yf.download(BRENT_SYMBOL, start=START_DATE, progress=False, auto_adjust=True)

    if wti.empty or brent.empty:
        raise RuntimeError("Yahoo returned empty data")

    wti = wti[["Close"]].rename(columns={"Close": "WTI_Close"})
    brent = brent[["Close"]].rename(columns={"Close": "Brent_Close"})

    df = wti.join(brent, how="inner")
    df = df.dropna().sort_index()
    return df

# ======================
# FEATURES
# ======================
def build_features(df):
    df = df.copy()

    df["WTI_Return"] = df["WTI_Close"].pct_change()
    df["Brent_Return"] = df["Brent_Close"].pct_change()

    df["Spread"] = df["Brent_Close"] - df["WTI_Close"]
    df["Spread_Change"] = df["Spread"].diff()

    for l in range(1, 6):
        df[f"WTI_Return_lag{l}"] = df["WTI_Return"].shift(l)
        df[f"Spread_Change_lag{l}"] = df["Spread_Change"].shift(l)

    # Target: next day direction (WTI)
    df["Target"] = (df["WTI_Return"].shift(-1) > 0).astype(int)

    df = df.dropna()
    return df

# ======================
# MODEL
# ======================
def train_model(df, features):
    X = df[features]
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], preds))

    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )
    final_model.fit(X, y)

    return final_model, float(np.mean(scores)), float(np.std(scores))

# ======================
# OUTPUT
# ======================
def write_txt(result):
    lines = []
    lines.append("===================================")
    lines.append("   OIL FORECAST – BRENT / WTI")
    lines.append("===================================")
    lines.append(f"Run time (UTC): {result['run_time']}")
    lines.append(f"Data date     : {result['data_date']}")
    lines.append("")
    lines.append(f"Model CV      : {result['cv_mean']:.2%} ± {result['cv_std']:.2%}")
    lines.append("")
    lines.append(f"Prob UP       : {result['prob_up']:.2%}")
    lines.append(f"Prob DOWN     : {result['prob_down']:.2%}")
    lines.append(f"Signal        : {result['signal']}")
    lines.append("===================================")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ======================
# MAIN
# ======================
def main():
    df_prices = load_prices()
    df = build_features(df_prices)

    if len(df) < MIN_ROWS:
        raise RuntimeError("Not enough data")

    feature_cols = [
        c for c in df.columns
        if c.startswith("WTI_Return_lag") or c.startswith("Spread_Change_lag")
    ]

    model, cv_mean, cv_std = train_model(df, feature_cols)

    last = df.iloc[-1:]
    prob_up = float(model.predict_proba(last[feature_cols])[0][1])
    prob_down = 1.0 - prob_up

    # Trend filter (WTI)
    trend_up = (
        df["WTI_Close"].iloc[-1]
        > df["WTI_Close"].rolling(50).mean().iloc[-1]
    )

    if prob_up >= PROB_TRADE_THRESHOLD and trend_up:
        signal = "UP"
    elif prob_down >= PROB_TRADE_THRESHOLD and not trend_up:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    result = {
        "run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_date": df.index[-1].date().isoformat(),
        "prob_up": prob_up,
        "prob_down": prob_down,
        "signal": signal,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }

    write_txt(result)

# ======================
# ENTRYPOINT
# ======================
if __name__ == "__main__":
    main()
