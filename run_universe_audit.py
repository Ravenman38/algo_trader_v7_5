#!/usr/bin/env python3
"""
Universe / market-cap audit for AlgoTrader v7.2.

This checks whether the ML training/prediction rows were actually in the intended
small/mid-cap range at the time of the row, using the historical market-cap proxy
created by train_probability_model.py --cap-mode proxy.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return ""


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return ""


def cap_bucket(cap):
    if pd.isna(cap):
        return "unknown"
    if cap < 300_000_000:
        return "micro"
    if cap < 2_000_000_000:
        return "small"
    if cap < 10_000_000_000:
        return "mid"
    if cap < 200_000_000_000:
        return "large"
    return "mega"


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing {path}. Run train-ml first.")
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def summarize_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "market_cap_proxy" not in df.columns:
        return pd.DataFrame([{"section": name, "warning": "market_cap_proxy missing; rerun train-ml with --cap-mode proxy"}])
    tmp = df.copy()
    tmp["cap_bucket_proxy"] = tmp.get("cap_bucket_proxy", tmp["market_cap_proxy"].map(cap_bucket))
    out = tmp.groupby("cap_bucket_proxy", as_index=False).agg(
        rows=("ticker", "size"),
        tickers=("ticker", "nunique"),
        median_market_cap_proxy=("market_cap_proxy", "median"),
        min_market_cap_proxy=("market_cap_proxy", "min"),
        max_market_cap_proxy=("market_cap_proxy", "max"),
    )
    out.insert(0, "section", name)
    out["row_share"] = out["rows"] / len(tmp)
    return out.sort_values("rows", ascending=False)


def main():
    train = load_csv("probability_training_dataset.csv")
    pred = load_csv("ml_probability_predictions.csv")

    parts = [
        summarize_rows(train, "training_rows"),
        summarize_rows(pred, "prediction_rows"),
    ]

    # Audit the most actionable subset: high-probability candidates by day.
    if "ml_prob_up_5pct" in pred.columns:
        p = pred.copy()
        p["rank"] = p.groupby("date")["ml_prob_up_5pct"].rank(method="first", ascending=False)
        top = p[p["rank"] <= 20].copy()
        parts.append(summarize_rows(top, "top20_daily_predictions"))

    summary = pd.concat(parts, ignore_index=True, sort=False)
    summary.to_csv("universe_audit_summary.csv", index=False)

    ticker_summary = pd.DataFrame()
    if "market_cap_proxy" in pred.columns:
        p = pred.copy()
        p["cap_bucket_proxy"] = p.get("cap_bucket_proxy", p["market_cap_proxy"].map(cap_bucket))
        ticker_summary = p.groupby("ticker", as_index=False).agg(
            prediction_rows=("date", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            median_market_cap_proxy=("market_cap_proxy", "median"),
            min_market_cap_proxy=("market_cap_proxy", "min"),
            max_market_cap_proxy=("market_cap_proxy", "max"),
            avg_ml_probability=("ml_prob_up_5pct", "mean"),
            avg_forward_return_5d=("forward_return_5d", "mean"),
        )
        ticker_summary["median_cap_bucket_proxy"] = ticker_summary["median_market_cap_proxy"].map(cap_bucket)
        ticker_summary = ticker_summary.sort_values("avg_forward_return_5d", ascending=False)
        ticker_summary.to_csv("universe_audit_by_ticker.csv", index=False)

    html = ["<html><head><title>Universe Audit</title>",
            "<style>body{font-family:Arial;margin:40px;} table{border-collapse:collapse;width:100%;font-size:13px;} th,td{border:1px solid #ddd;padding:7px;text-align:right;} th{background:#f3f3f3;} td:first-child,th:first-child{text-align:left;} .note{background:#fff8dc;padding:12px;border-left:4px solid #e0b000;}</style>",
            "</head><body>",
            "<h1>AlgoTrader v7.2 Universe / Market-Cap Audit</h1>",
            "<div class='note'>This audit checks whether rows used by the model were small/mid-cap at the historical decision date using a practical proxy: current shares outstanding estimate × historical close. This is still not a paid point-in-time database, but it is better than using today's index membership or today's market cap for all dates.</div>",
            "<h2>Summary by Cap Bucket</h2>"]
    html.append(summary.to_html(index=False, float_format=lambda x: f"{x:,.4f}"))
    if not ticker_summary.empty:
        html.append("<h2>Best Tickers by Average Forward 5-Day Return</h2>")
        show = ticker_summary.head(40).copy()
        for col in ["median_market_cap_proxy", "min_market_cap_proxy", "max_market_cap_proxy"]:
            show[col] = show[col].map(money)
        for col in ["avg_ml_probability", "avg_forward_return_5d"]:
            show[col] = show[col].map(pct)
        html.append(show.to_html(index=False))
    html.append("<h2>Files Saved</h2><ul><li>universe_audit_summary.csv</li><li>universe_audit_by_ticker.csv</li><li>trade_universe_cap_proxy_audit.csv, if cap proxy training was used</li></ul>")
    html.append("</body></html>")
    Path("universe_audit_report.html").write_text("\n".join(html))

    print("Saved universe_audit_report.html")
    print("Saved universe_audit_summary.csv")
    if not ticker_summary.empty:
        print("Saved universe_audit_by_ticker.csv")


if __name__ == "__main__":
    main()
