#!/usr/bin/env python3
"""
AlgoTrader v6 Data Integrity Checks

Run after training the ML model:
  python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2
  python run.py data-check

Inputs:
  probability_training_dataset.csv
  ml_probability_predictions.csv (optional)

Outputs:
  data_integrity_summary.csv
  data_integrity_ticker_flags.csv
  data_integrity_extreme_moves.csv
  data_integrity_report.html
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

TRAINING_FILE = "probability_training_dataset.csv"
PRED_FILE = "ml_probability_predictions.csv"


def load_data():
    if not os.path.exists(TRAINING_FILE):
        raise FileNotFoundError("probability_training_dataset.csv not found. Run train-ml first.")
    df = pd.read_csv(TRAINING_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    print("=" * 78)
    print("DATA INTEGRITY CHECKS")
    print("=" * 78)

    df = load_data()
    rows = []

    def add(check, status, detail):
        rows.append({"check": check, "status": status, "detail": detail})

    add("Rows loaded", "INFO", f"{len(df):,} rows")
    add("Tickers loaded", "INFO", f"{df['ticker'].nunique():,} tickers")
    add("Date range", "INFO", f"{df['date'].min().date()} to {df['date'].max().date()}")

    dupes = df.duplicated(["date", "ticker"]).sum()
    add("Duplicate ticker-date rows", "PASS" if dupes == 0 else "WARN", f"{dupes:,} duplicates")

    bad_close = ((df["close"] <= 0) | df["close"].isna()).sum() if "close" in df.columns else len(df)
    add("Valid close prices", "PASS" if bad_close == 0 else "FAIL", f"{bad_close:,} bad close values")

    # Extreme close values can be valid on adjusted data, but flag them for review.
    high_price = df[df["close"] > 1000][["date", "ticker", "close"]].copy() if "close" in df.columns else pd.DataFrame()
    add("Very high adjusted prices > $1,000", "WARN" if len(high_price) else "PASS", f"{len(high_price):,} rows")

    # Extreme forward returns may indicate split/ticker/data problems.
    ret_col = None
    for c in ["forward_return_5d", "future_5d_return", "ret_fwd_5d", "target_return_5d"]:
        if c in df.columns:
            ret_col = c
            break
    if ret_col:
        extreme = df[(df[ret_col].abs() > 1.0)][["date", "ticker", "close", ret_col]].copy()
        extreme = extreme.sort_values(ret_col, key=lambda s: s.abs(), ascending=False)
        add("Extreme 5-day returns > 100%", "WARN" if len(extreme) else "PASS", f"{len(extreme):,} rows")
        extreme.head(500).to_csv("data_integrity_extreme_moves.csv", index=False)
    else:
        extreme = pd.DataFrame()
        add("Forward return column found", "FAIL", "No forward return column found")

    # Missingness by feature.
    numeric = df.select_dtypes(include=[np.number])
    missing = numeric.isna().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0.02]
    add("Feature missingness", "WARN" if len(high_missing) else "PASS", f"{len(high_missing)} numeric columns >2% missing")

    # Ticker-level flags.
    flags = []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date")
        row = {"ticker": t, "rows": len(g), "start": g["date"].min().date(), "end": g["date"].max().date()}
        if "close" in g.columns:
            row["min_close"] = g["close"].min()
            row["max_close"] = g["close"].max()
            row["max_daily_abs_return"] = g["close"].pct_change().abs().max()
        if ret_col:
            row["max_abs_forward_return_5d"] = g[ret_col].abs().max()
        flags.append(row)
    flags = pd.DataFrame(flags)
    if not flags.empty:
        flags["flag_high_price"] = flags.get("max_close", 0) > 1000
        flags["flag_big_daily_move"] = flags.get("max_daily_abs_return", 0) > 0.75
        flags["flag_big_forward_move"] = flags.get("max_abs_forward_return_5d", 0) > 1.0
        flags = flags.sort_values(["flag_high_price", "flag_big_daily_move", "flag_big_forward_move", "max_close"], ascending=False)
        flags.to_csv("data_integrity_ticker_flags.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv("data_integrity_summary.csv", index=False)

    html = []
    html.append("<html><head><title>Data Integrity Report</title>")
    html.append("<style>body{font-family:Arial;margin:40px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:7px;text-align:right;} th{background:#f3f3f3;} td:first-child,th:first-child{text-align:left;}</style>")
    html.append("</head><body>")
    html.append("<h1>AlgoTrader v6 Data Integrity Report</h1>")
    html.append(f"<p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append("<h2>Summary Checks</h2>")
    html.append(summary.to_html(index=False))
    html.append("<h2>Top Ticker Flags</h2>")
    html.append(flags.head(50).to_html(index=False, float_format=lambda x: f"{x:,.4f}") if not flags.empty else "<p>No ticker flags.</p>")
    html.append("<h2>Extreme 5-Day Moves</h2>")
    html.append(extreme.head(50).to_html(index=False, float_format=lambda x: f"{x:,.4f}") if not extreme.empty else "<p>No extreme moves found.</p>")
    html.append("</body></html>")
    with open("data_integrity_report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(summary.to_string(index=False))
    print("\nSaved data_integrity_report.html")


if __name__ == "__main__":
    main()
