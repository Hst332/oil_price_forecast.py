#!/usr/bin/env python3
"""
oil_price_forecast.py
Ruhiger, robuster Öl-Forecast (Brent + WTI)
A-Variante: Trend + Wahrscheinlichkeit
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------
# Config
# -----------------------
START_DATE = "2010-01-01"
SYMBOL_BRENT = "BZ=F"   # Brent Crude
SYMBOL_WTI   = "CL=F"   # WTI Crude

PROB_TRADE_THRESHOLD = 0.57
OUTPUT_FILE = "oil_forecast_output.txt"

# -----------------------
# Load prices
# -----------------------
def load_prices():
    brent = yf.download(SYMBOL_BRENT, start=START_DATE, progress=False)["Close"]
    wti   = yf.download(SYMBOL_WTI, start=START_DATE, progress=False)["Close"]

    df = pd.concat([brent, wti], axis=1)
    df.columns = ["Brent_Close", "WTI_Close"]
    df = df.dropna()

    return df

# -----------------------
# Feature engineering
# -----------------------
def build_features(df):
    out = df.copy()

    # Returns
    out["Brent_Return"] = out["Brent_Close"].pct_change()
    out["WTI_Return"]   = out["WTI_Close"].pct_change()

    # Trend filter
    out["Brent_SMA50"] = out["Brent_Close"].rolling(50).mean()
    out["WTI_SMA50"]   = out["WTI_Close"].rolling(50).mean()

    # Trend regime
    out["Brent_Trend_Up"] = (out["Brent_Close"] > out["Brent_SMA50"]).astype(int)
    out["WTI_Trend_Up"]   = (out["WTI_Close"] > out["WTI_SMA50"]).astype(int)

    out = out.dropna()
    return out

# -----------------------
# Probability logic (robust, explainable)
# -----------------------
def compute_probability(df):
    last = df.iloc[-1]

    score = 0.5

    # Trend alignment
    if last["Brent_Trend_Up"]:
        score += 0.06
    else:
        score -= 0.06

    if last["WTI_Trend_Up"]:
        score += 0.06
    else:
        score -= 0.06

    # Momentum confirmation
    if last["Brent_Return"] > 0:
        score += 0.03
    if last["WTI_Return"] > 0:
        score += 0.03

    return max(0.0, min(1.0, score))

# -----------------------
# Output
# -----------------------
def write_output(result):
    lines = []
    lines.append("===================================")
    lines.append("        OIL PRICE FORECAST")
    lines.append("===================================")
    lines.append(f"Run time (UTC): {result['run_time']}")
    lines.append(f"Data date     : {result['data_date']}")
    lines.append("")
    lines.append("Sources:")
    lines.append("  Brent : Yahoo Finance (BZ=F)")
    lines.append("  WTI   : Yahoo Finance (CL=F)")
    lines.append("")
    lines.append(f"Brent Close : {result['brent_close']:.2f}")
    lines.append(f"WTI Close   : {result['wti_close']:.2f}")
    lines.append("")
    lines.append(f"Probability UP : {result['prob_up']:.2%}")
    lines.append(f"Signal        : {result['signal']}")
    lines.append("===================================")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))

# -----------------------
# Main
# -----------------------
def main():
    df = load_prices()
    df = build_features(df)

    prob_up = compute_probability(df)

    signal = (
        "TRADE"
        if prob_up >= PROB_TRADE_THRESHOLD
        else "NO_TRADE"
    )

    last = df.iloc[-1]

    result = {
        "run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_date": last.name.date().isoformat(),
        "brent_close": last["Brent_Close"],
        "wti_close": last["WTI_Close"],
        "prob_up": prob_up,
        "signal": signal,
    }

    write_output(result)

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    main()

