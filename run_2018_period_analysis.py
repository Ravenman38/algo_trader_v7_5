"""
2018+ Period Analysis Report
============================

Runs the portfolio backtest from 2018 to today, compares it to SPY, and
breaks performance down by calendar year and market regime.

Outputs:
  - period_analysis_overall.csv
  - period_analysis_yearly.csv
  - period_analysis_regimes.csv
  - period_analysis_trades_by_year.csv
  - period_analysis_equity.csv
  - period_analysis_equity_curve.png
  - period_analysis_drawdown.png
  - period_analysis_report.html

Run from the project root:
  python run_2018_period_analysis.py
"""

import math
import os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# Use a non-interactive backend so Colab can save charts without display issues.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_portfolio_backtest as pbt

START_DATE = "2018-01-01"
BENCHMARK = "SPY"
OUTPUT_DIR = Path(".")

# Market regimes to analyze. Dates are approximate, but useful for diagnosis.
REGIMES = [
    ("2018 correction", "2018-01-01", "2018-12-31"),
    ("2019 bull market", "2019-01-01", "2019-12-31"),
    ("2020 COVID crash", "2020-02-19", "2020-03-23"),
    ("2020-2021 recovery", "2020-03-24", "2021-12-31"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
    ("2023-2024 bull / AI rally", "2023-01-01", "2024-12-31"),
    ("2025-present", "2025-01-01", "2099-12-31"),
]


def pct(x):
    return round(float(x) * 100, 2) if pd.notna(x) else np.nan


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


def perf_metrics(equity: pd.Series) -> dict:
    """Compute basic performance metrics from an equity curve."""
    equity = equity.dropna().astype(float)
    if len(equity) < 2:
        return {
            "start": np.nan,
            "end": np.nan,
            "total_return_pct": np.nan,
            "cagr_pct": np.nan,
            "volatility_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
        }

    daily = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0
    vol = daily.std() * math.sqrt(252) if len(daily) > 1 else np.nan
    dd = max_drawdown(equity)
    sharpe = daily.mean() / daily.std() * math.sqrt(252) if len(daily) > 1 and daily.std() > 0 else np.nan
    downside = daily[daily < 0]
    sortino = daily.mean() / downside.std() * math.sqrt(252) if len(downside) > 1 and downside.std() > 0 else np.nan
    calmar = cagr / abs(dd) if pd.notna(dd) and dd < 0 else np.nan

    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "total_return_pct": pct(total_return),
        "cagr_pct": pct(cagr),
        "volatility_pct": pct(vol),
        "max_drawdown_pct": pct(dd),
        "sharpe": round(float(sharpe), 2) if pd.notna(sharpe) else np.nan,
        "sortino": round(float(sortino), 2) if pd.notna(sortino) else np.nan,
        "calmar": round(float(calmar), 2) if pd.notna(calmar) else np.nan,
    }


def slice_equity(equity: pd.Series, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    s = equity[(equity.index >= start_ts) & (equity.index <= end_ts)].copy()
    return s


def make_spy_equity(spy: pd.DataFrame, strategy_equity: pd.Series) -> pd.Series:
    spy_close = spy["Close"].copy()
    spy_close.index = pd.to_datetime(spy_close.index)
    spy_close = spy_close.loc[(spy_close.index >= strategy_equity.index[0]) & (spy_close.index <= strategy_equity.index[-1])]
    spy_close = spy_close.reindex(strategy_equity.index, method="ffill").dropna()
    initial = float(strategy_equity.loc[spy_close.index[0]])
    return spy_close / spy_close.iloc[0] * initial


def build_overall(strategy: pd.Series, spy_eq: pd.Series, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, eq in [("Strategy", strategy), ("SPY", spy_eq)]:
        row = {"asset": name}
        row.update(perf_metrics(eq))
        if name == "Strategy":
            row["trades"] = int(len(trades))
            if not trades.empty and "net_return_pct" in trades.columns:
                rets = trades["net_return_pct"].astype(float) / 100.0
                wins = rets[rets > 0]
                losses = rets[rets <= 0]
                row["win_rate_pct"] = pct(len(wins) / len(rets)) if len(rets) else np.nan
                row["avg_trade_pct"] = pct(rets.mean()) if len(rets) else np.nan
                gross_profit = trades.loc[trades["pnl_$"] > 0, "pnl_$"].sum() if "pnl_$" in trades else np.nan
                gross_loss = -trades.loc[trades["pnl_$"] <= 0, "pnl_$"].sum() if "pnl_$" in trades else np.nan
                row["profit_factor"] = round(float(gross_profit / gross_loss), 2) if gross_loss and gross_loss > 0 else np.nan
        else:
            row["trades"] = 0
            row["win_rate_pct"] = np.nan
            row["avg_trade_pct"] = np.nan
            row["profit_factor"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 2:
        strat = df.iloc[0]
        bench = df.iloc[1]
        diff = {"asset": "Strategy minus SPY"}
        for col in ["total_return_pct", "cagr_pct", "volatility_pct", "max_drawdown_pct", "sharpe", "sortino", "calmar"]:
            diff[col] = round(float(strat[col]) - float(bench[col]), 2) if pd.notna(strat[col]) and pd.notna(bench[col]) else np.nan
        diff["trades"] = int(strat.get("trades", 0))
        diff["win_rate_pct"] = strat.get("win_rate_pct", np.nan)
        diff["avg_trade_pct"] = strat.get("avg_trade_pct", np.nan)
        diff["profit_factor"] = strat.get("profit_factor", np.nan)
        df = pd.concat([df, pd.DataFrame([diff])], ignore_index=True)
    return df


def build_yearly(strategy: pd.Series, spy_eq: pd.Series) -> pd.DataFrame:
    years = sorted(set(strategy.index.year) | set(spy_eq.index.year))
    rows = []
    for y in years:
        s = slice_equity(strategy, f"{y}-01-01", f"{y}-12-31")
        b = slice_equity(spy_eq, f"{y}-01-01", f"{y}-12-31")
        if len(s) < 2 or len(b) < 2:
            continue
        s_ret = s.iloc[-1] / s.iloc[0] - 1.0
        b_ret = b.iloc[-1] / b.iloc[0] - 1.0
        rows.append({
            "year": y,
            "strategy_return_pct": pct(s_ret),
            "spy_return_pct": pct(b_ret),
            "excess_return_pct": pct(s_ret - b_ret),
            "strategy_max_drawdown_pct": pct(max_drawdown(s)),
            "spy_max_drawdown_pct": pct(max_drawdown(b)),
            "beat_spy": bool(s_ret > b_ret),
        })
    return pd.DataFrame(rows)


def build_regimes(strategy: pd.Series, spy_eq: pd.Series, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trade_dates = pd.to_datetime(trades["entry_date"]) if not trades.empty and "entry_date" in trades.columns else pd.Series(dtype="datetime64[ns]")
    for name, start, end in REGIMES:
        s = slice_equity(strategy, start, end)
        b = slice_equity(spy_eq, start, end)
        if len(s) < 2 or len(b) < 2:
            continue
        sm = perf_metrics(s)
        bm = perf_metrics(b)
        s_ret = s.iloc[-1] / s.iloc[0] - 1.0
        b_ret = b.iloc[-1] / b.iloc[0] - 1.0
        n_trades = int(((trade_dates >= pd.Timestamp(start)) & (trade_dates <= pd.Timestamp(end))).sum()) if len(trade_dates) else 0
        rows.append({
            "period": name,
            "start": sm["start"],
            "end": sm["end"],
            "strategy_return_pct": sm["total_return_pct"],
            "spy_return_pct": bm["total_return_pct"],
            "excess_return_pct": pct(s_ret - b_ret),
            "strategy_cagr_pct": sm["cagr_pct"],
            "spy_cagr_pct": bm["cagr_pct"],
            "strategy_max_drawdown_pct": sm["max_drawdown_pct"],
            "spy_max_drawdown_pct": bm["max_drawdown_pct"],
            "strategy_sharpe": sm["sharpe"],
            "spy_sharpe": bm["sharpe"],
            "trades": n_trades,
            "beat_spy": bool(s_ret > b_ret),
        })
    return pd.DataFrame(rows)


def build_trades_by_year(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "entry_date" not in trades.columns:
        return pd.DataFrame()
    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["year"] = t["entry_date"].dt.year
    rows = []
    for year, g in t.groupby("year"):
        rets = g["net_return_pct"].astype(float) / 100.0 if "net_return_pct" in g else pd.Series(dtype=float)
        wins = rets[rets > 0]
        gross_profit = g.loc[g["pnl_$"] > 0, "pnl_$"].sum() if "pnl_$" in g else np.nan
        gross_loss = -g.loc[g["pnl_$"] <= 0, "pnl_$"].sum() if "pnl_$" in g else np.nan
        rows.append({
            "year": int(year),
            "trades": int(len(g)),
            "win_rate_pct": pct(len(wins) / len(rets)) if len(rets) else np.nan,
            "avg_trade_pct": pct(rets.mean()) if len(rets) else np.nan,
            "best_trade_pct": pct(rets.max()) if len(rets) else np.nan,
            "worst_trade_pct": pct(rets.min()) if len(rets) else np.nan,
            "profit_factor": round(float(gross_profit / gross_loss), 2) if gross_loss and gross_loss > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def save_charts(strategy: pd.Series, spy_eq: pd.Series) -> tuple[str, str]:
    # Equity curve
    plt.figure(figsize=(11, 6))
    plt.plot(strategy.index, strategy.values, label="Strategy")
    plt.plot(spy_eq.index, spy_eq.values, label="SPY")
    plt.title("Equity Curve: Strategy vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    equity_png = "period_analysis_equity_curve.png"
    plt.savefig(equity_png, dpi=140)
    plt.close()

    # Drawdown
    plt.figure(figsize=(11, 6))
    plt.plot(strategy.index, (strategy / strategy.cummax() - 1.0) * 100, label="Strategy")
    plt.plot(spy_eq.index, (spy_eq / spy_eq.cummax() - 1.0) * 100, label="SPY")
    plt.title("Drawdown: Strategy vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    drawdown_png = "period_analysis_drawdown.png"
    plt.savefig(drawdown_png, dpi=140)
    plt.close()
    return equity_png, drawdown_png


def table_html(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h2>{title}</h2><p>No data.</p>"
    return f"<h2>{title}</h2>" + df.to_html(index=False, border=0, classes="dataframe")


def write_html_report(overall, yearly, regimes, trades_by_year, equity_png, drawdown_png):
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>2018+ Strategy Period Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 32px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f4f4f4; }}
    .note {{ color: #555; }}
    img {{ max-width: 100%; margin: 10px 0 24px 0; border: 1px solid #eee; }}
  </style>
</head>
<body>
  <h1>2018+ Strategy Period Analysis</h1>
  <p class="note">Generated by <code>run_2018_period_analysis.py</code>. Includes commissions and the portfolio construction rules in your current backtester.</p>

  <h2>Equity Curve</h2>
  <img src="{equity_png}" alt="Equity curve">

  <h2>Drawdown</h2>
  <img src="{drawdown_png}" alt="Drawdown chart">

  {table_html(overall, "Overall Performance")}
  {table_html(yearly, "Year-by-Year Performance")}
  {table_html(regimes, "Market Regime Performance")}
  {table_html(trades_by_year, "Trade Statistics by Year")}

  <h2>How to read this</h2>
  <p>Focus on whether the strategy beats SPY across multiple years and regimes, not just overall. Watch for years where the strategy underperforms, large drawdowns, and whether results rely heavily on one unusually strong period.</p>
  <p class="note">This is research output, not investment advice. Before using real capital, test next-day execution, slippage, survivorship bias, and paper trading results.</p>
</body>
</html>
"""
    Path("period_analysis_report.html").write_text(html, encoding="utf-8")


def main():
    print("=" * 72)
    print("2018+ PERIOD ANALYSIS: STRATEGY VS SPY")
    print("=" * 72)

    today = dt.date.today()
    start = dt.date.fromisoformat(START_DATE)
    # Add a little buffer so the simulator starts no later than 2018.
    pbt.BACKTEST_YEARS = math.ceil((today - start).days / 365.25) + 1

    print(f"Requested start date: {START_DATE}")
    print(f"Using lookback years in backtester: {pbt.BACKTEST_YEARS}")
    print("Running portfolio simulation. This may take several minutes in Colab...\n")

    price_data, spy, market_caps = pbt.prepare_data()
    result = pbt.simulate_portfolio(price_data, spy, market_caps)

    equity_df = result.equity_curve.copy()
    trades = result.trades.copy()
    if equity_df.empty:
        raise RuntimeError("Backtest produced no equity curve.")

    equity_df["date"] = pd.to_datetime(equity_df["date"])
    equity_df = equity_df[equity_df["date"] >= pd.Timestamp(START_DATE)].copy()
    equity_df = equity_df.sort_values("date")
    strategy = equity_df.set_index("date")["equity"].astype(float)

    if len(strategy) < 2:
        raise RuntimeError("Not enough strategy equity records after 2018-01-01.")

    spy_eq = make_spy_equity(spy, strategy)
    common_idx = strategy.index.intersection(spy_eq.index)
    strategy = strategy.loc[common_idx]
    spy_eq = spy_eq.loc[common_idx]

    if not trades.empty and "entry_date" in trades.columns:
        trades = trades[pd.to_datetime(trades["entry_date"]) >= pd.Timestamp(START_DATE)].copy()

    overall = build_overall(strategy, spy_eq, trades)
    yearly = build_yearly(strategy, spy_eq)
    regimes = build_regimes(strategy, spy_eq, trades)
    trades_by_year = build_trades_by_year(trades)

    combined_equity = pd.DataFrame({
        "date": strategy.index,
        "strategy_equity": strategy.values,
        "spy_equity": spy_eq.values,
        "strategy_drawdown_pct": (strategy / strategy.cummax() - 1.0).values * 100,
        "spy_drawdown_pct": (spy_eq / spy_eq.cummax() - 1.0).values * 100,
    })

    equity_png, drawdown_png = save_charts(strategy, spy_eq)

    overall.to_csv("period_analysis_overall.csv", index=False)
    yearly.to_csv("period_analysis_yearly.csv", index=False)
    regimes.to_csv("period_analysis_regimes.csv", index=False)
    trades_by_year.to_csv("period_analysis_trades_by_year.csv", index=False)
    combined_equity.to_csv("period_analysis_equity.csv", index=False)
    result.trades.to_csv("period_analysis_trades.csv", index=False)
    write_html_report(overall, yearly, regimes, trades_by_year, equity_png, drawdown_png)

    print("\n" + "=" * 72)
    print("OVERALL PERFORMANCE")
    print("=" * 72)
    print(overall.to_string(index=False))

    print("\n" + "=" * 72)
    print("YEAR-BY-YEAR PERFORMANCE")
    print("=" * 72)
    print(yearly.to_string(index=False))

    print("\n" + "=" * 72)
    print("MARKET REGIME PERFORMANCE")
    print("=" * 72)
    print(regimes.to_string(index=False))

    print("\nSaved:")
    for f in [
        "period_analysis_overall.csv",
        "period_analysis_yearly.csv",
        "period_analysis_regimes.csv",
        "period_analysis_trades_by_year.csv",
        "period_analysis_equity.csv",
        "period_analysis_trades.csv",
        equity_png,
        drawdown_png,
        "period_analysis_report.html",
    ]:
        print(f"  {f}")

    print("\nOpen period_analysis_report.html for the full report.")


if __name__ == "__main__":
    main()
