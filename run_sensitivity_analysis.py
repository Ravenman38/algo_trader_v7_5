#!/usr/bin/env python3
"""
AlgoTrader v6 Sensitivity Analysis

Tests whether the ML strategy still works when reasonable assumptions change:
  - holding period / rebalance frequency
  - probability threshold
  - max position size
  - slippage

Run after:
  python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2

Then:
  python run.py sensitivity

Outputs:
  sensitivity_analysis_summary.csv
  sensitivity_analysis_report.html
  sensitivity_cagr_heatmap.png
  sensitivity_sharpe_heatmap.png
"""

import itertools
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import run_walkforward_comparison as wf


HOLD_DAYS_GRID = [3, 5, 7, 10]
PROB_THRESHOLD_GRID = [0.25, 0.30, 0.35, 0.40]
MAX_POSITION_GRID = [0.10, 0.15, 0.20]
SLIPPAGE_BPS_GRID = [5, 10, 20]


def get_spy(start, end):
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if spy.empty:
        return pd.DataFrame()
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    spy = spy.reset_index()
    spy.columns = [str(c).lower() for c in spy.columns]
    date_col = "date" if "date" in spy.columns else "datetime"
    spy = spy[[date_col, "close"]].rename(columns={date_col: "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy["equity"] = wf.INITIAL_CAPITAL * spy["close"] / spy["close"].iloc[0]
    spy["asset"] = "SPY"
    return spy[["date", "equity", "asset"]]


def run_one(ml, hold_days, threshold, max_position, slippage_bps):
    old = (wf.HOLD_DAYS, wf.PROB_THRESHOLD, wf.MAX_POSITION_PCT, wf.SLIPPAGE_BPS)
    wf.HOLD_DAYS = hold_days
    wf.PROB_THRESHOLD = threshold
    wf.MAX_POSITION_PCT = max_position
    wf.SLIPPAGE_BPS = slippage_bps
    try:
        eq, tr = wf.simulate(ml, "ml_score", "ML Strategy")
        if eq.empty:
            return None
        met = wf.metrics_for_equity(eq, "ML Strategy")
        wins = tr[tr["net_pnl"] > 0] if not tr.empty else pd.DataFrame()
        losses = tr[tr["net_pnl"] <= 0] if not tr.empty else pd.DataFrame()
        met.update({
            "hold_days": hold_days,
            "prob_threshold": threshold,
            "max_position_pct": max_position,
            "slippage_bps": slippage_bps,
            "trades": len(tr),
            "win_rate_pct": (tr["net_pnl"] > 0).mean() * 100 if len(tr) else np.nan,
            "avg_trade_pct": tr["net_return_pct"].mean() if len(tr) else np.nan,
            "profit_factor": wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()) if len(losses) and losses["net_pnl"].sum() != 0 else np.nan,
            "total_commissions": tr["commission"].sum() if len(tr) else np.nan,
        })
        return met
    finally:
        wf.HOLD_DAYS, wf.PROB_THRESHOLD, wf.MAX_POSITION_PCT, wf.SLIPPAGE_BPS = old


def make_heatmap(df, value, fname, title):
    # Base view: hold_days x threshold using default max_position=15%, slippage=5 bps
    sub = df[(df["max_position_pct"] == 0.15) & (df["slippage_bps"] == 5)].copy()
    if sub.empty:
        return None
    pivot = sub.pivot(index="hold_days", columns="prob_threshold", values=value)
    plt.figure(figsize=(9, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label=value)
    plt.xticks(range(len(pivot.columns)), [f"{x:.0%}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Probability Threshold")
    plt.ylabel("Holding Days")
    plt.title(title)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                plt.text(j, i, f"{val:.1f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()
    return fname


def main():
    print("=" * 78)
    print("SENSITIVITY ANALYSIS")
    print("=" * 78)

    ml, _ = wf.load_ml_and_training_data()
    start = ml["date"].min()
    end = ml["date"].max()
    print(f"Testing window: {start.date()} to {end.date()}")

    rows = []
    combos = list(itertools.product(HOLD_DAYS_GRID, PROB_THRESHOLD_GRID, MAX_POSITION_GRID, SLIPPAGE_BPS_GRID))
    for i, (h, p, m, s) in enumerate(combos, 1):
        print(f"{i}/{len(combos)}: hold={h}, threshold={p:.0%}, max_pos={m:.0%}, slippage={s}bps")
        res = run_one(ml, h, p, m, s)
        if res is not None:
            rows.append(res)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No sensitivity results created.")

    out = out.sort_values(["sharpe", "cagr_pct"], ascending=False)
    out.to_csv("sensitivity_analysis_summary.csv", index=False)

    cagr_chart = make_heatmap(out, "cagr_pct", "sensitivity_cagr_heatmap.png", "CAGR Sensitivity (max pos 15%, slippage 5bps)")
    sharpe_chart = make_heatmap(out, "sharpe", "sensitivity_sharpe_heatmap.png", "Sharpe Sensitivity (max pos 15%, slippage 5bps)")

    baseline = out[(out["hold_days"] == 5) & (out["prob_threshold"] == 0.30) & (out["max_position_pct"] == 0.15) & (out["slippage_bps"] == 5)]
    robust = out[(out["cagr_pct"] > 15) & (out["sharpe"] > 1) & (out["max_drawdown_pct"] > -45)]

    html = []
    html.append("<html><head><title>Sensitivity Analysis</title>")
    html.append("<style>body{font-family:Arial;margin:40px;} table{border-collapse:collapse;width:100%;font-size:13px;} th,td{border:1px solid #ddd;padding:7px;text-align:right;} th{background:#f3f3f3;} td:first-child,th:first-child{text-align:left;} img{max-width:100%;border:1px solid #ddd;margin:20px 0;}</style>")
    html.append("</head><body>")
    html.append("<h1>AlgoTrader v6 Sensitivity Analysis</h1>")
    html.append(f"<p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append(f"<p><b>Robust configurations:</b> {len(robust)} of {len(out)} passed CAGR > 15%, Sharpe > 1, max drawdown better than -45%.</p>")
    html.append("<h2>Baseline Configuration</h2>")
    html.append(baseline.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not baseline.empty else "<p>Baseline not found.</p>")
    html.append("<h2>Top 25 Configurations</h2>")
    html.append(out.head(25).to_html(index=False, float_format=lambda x: f"{x:,.2f}"))
    html.append("<h2>Worst 25 Configurations</h2>")
    html.append(out.tail(25).to_html(index=False, float_format=lambda x: f"{x:,.2f}"))
    for chart in [cagr_chart, sharpe_chart]:
        if chart:
            html.append(f"<h2>{chart}</h2><img src='{chart}' />")
    html.append("</body></html>")
    with open("sensitivity_analysis_report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print("\nTop configurations:")
    print(out.head(10).to_string(index=False))
    print("\nSaved sensitivity_analysis_report.html")


if __name__ == "__main__":
    main()
