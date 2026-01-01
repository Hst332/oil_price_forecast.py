#!/usr/bin/env python3
"""
CODE A – ENERGY FORECAST
Natural Gas + Oil (Brent / WTI)

Ruhig. Robust. Professionell.
Gas = ML
Oil = Rule-based
One TXT output
"""

# =======================
# IMPORTS
# =======================
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# =======================
# CONFIG
# =======================
START_DATE_GAS = "2014-01-01"
START_DATE_OIL = "2015-01-01"

GAS_SYMBOL = "NG=F"
SYMBOL_BRENT = "BZ=F"
SYMBOL_WTI = "CL=F"

OUT_TXT = "energy_forecast_output.txt"

UP_THRESHOLD = 0.60
DOWN_THRESHOLD = 0.40

# =======================
# -------- GAS ----------
# =======================
def load_gas_prices():
    df = yf.download(
        GAS_SYMBOL,
        start=START_DATE_GAS,
        auto_adjust=True,
        progress=False,
    )
    df = df[["Close"]].rename(columns={"Close": "Gas_Close"})
    df.dropna(inplace=True)
    return df


def load_eia_storage():
    try:
        s = pd.read_csv("eia_storage.csv", parse_dates=["Date"])
        s.sort_values("Date", inplace=True)
        return s
    except Exception:
        return None


def build_gas_features(price_df, storage_df):
    df = price_df.copy()

    df["ret"] = df["Gas_Close"].pct_change()
    df["trend_5"] = df["Gas_Close"].pct_change(5)
    df["trend_20"] = df["Gas_Close"].pct_change(20)
    df["vol_10"] = df["ret"].rolling(10).std()
    df["Target"] = (df["ret"].shift(-1) > 0).astype(int)

    if storage_df is not None:
        storage_df["surprise"] = (
            storage_df["Storage"] - storage_df["FiveYearAvg"]
        )
        storage_df["surprise_z"] = (
            (storage_df["surprise"] - storage_df["surprise"].rolling(52).mean())
            / storage_df["surprise"].rolling(52).std()
        )
        storage_df = storage_df[["Date", "surprise_z"]]
        df = df.merge(
            storage_df,
            left_index=True,
            right_on="Date",
            how="left",
        )
        df["surprise_z"].ffill(inplace=True)
        df.set_index("Date", inplace=True)
    else:
        df["surprise_z"] = 0.0

    df.dropna(inplace=True)
    return df


def train_gas_model(df):
    features = ["trend_5", "trend_20", "vol_10", "surprise_z"]
    X = df[features]
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=5)
    acc = []

    for tr, te in tscv.split(X):
        m = LogisticRegression(max_iter=200)
        m.fit(X.iloc[tr], y.iloc[tr])
        acc.append(
            accuracy_score(y.iloc[te], m.predict(X.iloc[te]))
        )

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return model, features, float(np.mean(acc)), float(np.std(acc))


# =======================
# -------- OIL ----------
# =======================
def load_oil_prices():
    brent = yf.download(
        SYMBOL_BRENT,
        start=START_DATE_OIL,
        progress=False,
    )
    wti = yf.download(
        SYMBOL_WTI,
        start=START_DATE_OIL,
        progress=False,
    )

    df = pd.DataFrame(index=brent.index)
    df["Brent_Close"] = brent["Close"]
    df["WTI_Close"] = wti["Close"]
    df.dropna(inplace=True)
    return df


def build_oil_signal(df):
    df = df.copy()

    df["Brent_Trend"] = (
        df["Brent_Close"]
        > df["Brent_Close"].rolling(20).mean()
    )
    df["WTI_Trend"] = (
        df["WTI_Close"]
        > df["WTI_Close"].rolling(20).mean()
    )

    df["Spread"] = df["Brent_Close"] - df["WTI_Close"]
    df["Spread_Z"] = (
        (df["Spread"] - df["Spread"].rolling(60).mean())
        / df["Spread"].rolling(60).std()
    )

    df.dropna(inplace=True)
    last = df.iloc[-1]

    prob_up = 0.50

    if last["Brent_Trend"] and last["WTI_Trend"]:
        prob_up += 0.07

    if last["Spread_Z"] > 0.5:
        prob_up += 0.03
    elif last["Spread_Z"] < -0.5:
        prob_up -= 0.03

    prob_up = max(0.0, min(1.0, prob_up))

    if prob_up >= 0.57:
        signal = "UP"
    elif prob_up <= 0.43:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    return {
        "date": last.name.date().isoformat(),
        "brent": float(last["Brent_Close"]),
        "wti": float(last["WTI_Close"]),
        "spread": float(last["Spread"]),
        "prob_up": prob_up,
        "prob_down": 1.0 - prob_up,
        "signal": signal,
    }


# =======================
# OUTPUT
# =======================
def write_output(gas, oil):
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("===================================\n")
        f.write("   ENERGY FORECAST – CODE A\n")
        f.write("===================================\n")
        f.write(
            f"Run time (UTC): "
            f"{datetime.utcnow():%Y-%m-%d %H:%M:%S UTC}\n\n"
        )

        f.write("--------- NATURAL GAS ---------\n")
        f.write(f"Data date : {gas['date']}\n")
        f.write(
            f"Model CV  : "
            f"{gas['cv_mean']:.2%} ± {gas['cv_std']:.2%}\n"
        )
        f.write(f"Prob UP   : {gas['prob_up']:.2%}\n")
        f.write(f"Prob DOWN : {gas['prob_down']:.2%}\n")
        f.write(f"Signal    : {gas['signal']}\n\n")

        f.write("--------- OIL (BRENT / WTI) ---------\n")
        f.write(f"Data date     : {oil['date']}\n")
        f.write(f"Brent Close   : {oil['brent']:.2f}\n")
        f.write(f"WTI Close     : {oil['wti']:.2f}\n")
        f.write(f"Brent–WTI Spd : {oil['spread']:.2f}\n")
        f.write(f"Prob UP       : {oil['prob_up']:.2%}\n")
        f.write(f"Prob DOWN     : {oil['prob_down']:.2%}\n")
        f.write(f"Signal        : {oil['signal']}\n")
        f.write("===================================\n")


# =======================
# MAIN
# =======================
def main():
    gas_prices = load_gas_prices()
    storage = load_eia_storage()
    gas_df = build_gas_features(gas_prices, storage)

    model, features, cv_mean, cv_std = train_gas_model(gas_df)

    last = gas_df.iloc[-1:]
    prob_up = model.predict_proba(last[features])[0][1]

    gas_signal = (
        "UP" if prob_up >= UP_THRESHOLD
        else "DOWN" if prob_up <= DOWN_THRESHOLD
        else "NO_TRADE"
    )

    gas_res = {
        "date": last.index[0].date().isoformat(),
        "prob_up": prob_up,
        "prob_down": 1.0 - prob_up,
        "signal": gas_signal,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }

    oil_df = load_oil_prices()
    oil_res = build_oil_signal(oil_df)

    write_output(gas_res, oil_res)
    print("[OK] Energy forecast written")


if __name__ == "__main__":
    main()
