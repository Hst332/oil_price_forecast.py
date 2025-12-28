#!/usr/bin/env python3
"""
oil_price_forecast.py
CODE A – ruhig, robust, professionell

Brent + WTI + Spread
TXT Output (wird jedes Mal überschrieben)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
START_DATE = "2015-01-01"
SYMBOL_BRENT = "BZ=F"
SYMBOL_WTI = "CL=F"
OUTPUT_TXT = "oil_price_forecast.txt"

# =========================
# LOAD DATA
# =========================
def load_prices():
    brent = yf.download(SYMBOL_BRENT, start=START_DATE, progress=False)
    wti = yf.download(SYMBOL_WTI, start=START_DATE, progress=False)

    if brent.empty or wti.empty:
        raise RuntimeError("Yahoo data download failed")

    df = pd.DataFrame(index=brent.index)
    df["Brent_Close"] = brent["Close"]
    df["WTI_Close"] = wti["Close"]
    df = df.dropna()

    return df

# =========================
# SIGNAL LOGIC – CODE A
# =========================
def build_signal(df: pd.DataFrame):
    df = df.copy()

    # Trendfilter
    df["Brent_Trend"] = df["Brent_Close"] > df["Brent_Close"].rolling(20).mean()
    df["WTI_Trend"] = df["WTI_Close"] > df["WTI_Close"].rolling(20).mean()

    # Spread
    df["Spread"] = df["Brent_Close"] - df["WTI_Close"]
    df["Spread_Z"] = (
        (df["Spread"] - df["Spread"].rolling(60).mean())
        / df["Spread"].rolling(60).std()
    )

    df = df.dropna()
    last = df.iloc[-1]

    prob_up = 0.50

    if last["Brent_Trend"] and last["WTI_Trend"]:
        prob_up += 0.07

    if last["Spread_Z"] > 0.5:
        prob_up += 0.03
    elif last["Spread_Z"] < -0.5:
        prob_up -= 0.03

    prob_up = max(0.0, min(1.0, prob_up))
    prob_down = 1.0 - prob_up

    if prob_up >= 0.57:
        signal = "UP"
    elif prob_up <= 0.43:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    return {
        "run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_date": last.name.date().isoformat(),
        "prob_up": prob_up,
        "prob_down": prob_down,
        "signal": signal,
        "brent": float(last["Brent_Close"]),
        "wti": float(last["WTI_Close"]),
        "spread": float(last["Spread"]),
    }

# =========================
# OUTPUT
# =========================
def write_output_txt(result: dict):
    text = f"""===================================
      OIL FORECAST – CODE A
===================================
Run time (UTC): {result['run_time']}
Data date     : {result['data_date']}

Brent Close   : {result['brent']:.2f}
WTI Close     : {result['wti']:.2f}
Brent–WTI Spd : {result['spread']:.2f}

Prob UP       : {result['prob_up']:.2%}
Prob DOWN     : {result['prob_down']:.2%}
Signal        : {result['signal']}
===================================
"""

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(text)

# =========================
# MAIN
# =========================
def main():
    df = load_prices()
    result = build_signal(df)
    write_output_txt(result)
    print("[OK] oil_price_forecast.txt created")

if __name__ == "__main__":
    main()
