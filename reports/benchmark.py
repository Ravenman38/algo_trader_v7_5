"""
Benchmark report for AlgoTrader portfolio backtest.

Run from the repo root after run_portfolio_backtest.py has created:
  - portfolio_backtest_equity.csv
  - portfolio_backtest_trades.csv

Outputs:
  - benchmark_report_summary.csv
  - benchmark_report_monthly_returns.csv
  - benchmark_equity_curve.png
  - benchmark_drawdown.png
  - benchmark_rolling_12m_returns.png
  - benchmark_trade_distribution.png
  - benchmark_report.html
"""

import os
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit("Missing dependency: yfinance. Run: pip install yfinance") from exc

EQUITY_FILE = "portfolio_backtest_equity.csv"
TRADES_FILE = "portfolio_backtest_trades.csv"
BENCHMARK = "SPY"
RISK_FREE_RATE = 0.0
TRADING_DAYS = 252


@dataclass
class PerfStats:
    name: str
    start_date: str
    end_date: str
    start_value: float
    end_value: float
    total_return_pct: float
    cagr_pct: float
    annual_vol_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    calmar: float


def _find_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "Date", "timestamp", "Timestamp"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _find_equity_col(df: pd.DataFrame) -> str:
    candidates = ["equity", "portfolio_value", "ending_equity", "value", "capital", "balance"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("Could not find an equity/value column in portfolio_backtest_equity.csv")
    return numeric_cols[-1]


def load_strategy_equity(path: str = EQUITY_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run python run_portfolio_backtest.py first."
        )

    df = pd.read_csv(path)
    date_col = _find_date_col(df)
    equity_col = _find_equity_col(df)

    out = df[[date_col, equity_col]].copy()
    out.columns = ["date", "strategy"]
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna().sort_values("date").drop_duplicates("date")
    out = out.set_index("date")
    out["strategy"] = pd.to_numeric(out["strategy"], errors="coerce")
    out = out.dropna()
    return out


def load_spy_equity(start: pd.Timestamp, end: pd.Timestamp, initial_value: float) -> pd.DataFrame:
    raw = yf.download(
        BENCHMARK,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError("Could not download SPY data from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    spy = pd.DataFrame({"spy": close})
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy = spy.dropna()
    spy["spy"] = initial_value * spy["spy"] / spy["spy"].iloc[0]
    return spy


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def calc_stats(name: str, equity: pd.Series) -> PerfStats:
    equity = equity.dropna()
    rets = daily_returns(equity)
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    total_return = end_value / start_value - 1.0
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else np.nan
    annual_vol = rets.std() * math.sqrt(TRADING_DAYS) if len(rets) else np.nan
    excess = rets - RISK_FREE_RATE / TRADING_DAYS
    sharpe = excess.mean() / excess.std() * math.sqrt(TRADING_DAYS) if excess.std() else np.nan
    downside = rets[rets < 0]
    sortino = excess.mean() / downside.std() * math.sqrt(TRADING_DAYS) if len(downside) and downside.std() else np.nan
    max_dd = drawdown(equity).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    return PerfStats(
        name=name,
        start_date=equity.index[0].strftime("%Y-%m-%d"),
        end_date=equity.index[-1].strftime("%Y-%m-%d"),
        start_value=start_value,
        end_value=end_value,
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        annual_vol_pct=annual_vol * 100,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd * 100,
        calmar=calmar,
    )


def stats_to_frame(stats) -> pd.DataFrame:
    return pd.DataFrame([s.__dict__ for s in stats]).round(4)


def monthly_returns_table(equity: pd.Series) -> pd.DataFrame:
    monthly = equity.resample("M").last().pct_change().dropna()
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.strftime("%b")
    pivot = table.pivot(index="year", columns="month", values="return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=month_order)
    return (pivot * 100).round(2)


def save_charts(combined: pd.DataFrame, trades: Optional[pd.DataFrame]) -> None:
    norm = combined / combined.iloc[0] * 100000

    plt.figure(figsize=(11, 6))
    plt.plot(norm.index, norm["strategy"], label="Strategy")
    plt.plot(norm.index, norm["spy"], label="SPY")
    plt.title("Equity Curve: Strategy vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_equity_curve.png", dpi=150)
    plt.close()

    dd = combined.apply(drawdown) * 100
    plt.figure(figsize=(11, 6))
    plt.plot(dd.index, dd["strategy"], label="Strategy")
    plt.plot(dd.index, dd["spy"], label="SPY")
    plt.title("Drawdown: Strategy vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_drawdown.png", dpi=150)
    plt.close()

    rolling = combined.pct_change(252) * 100
    plt.figure(figsize=(11, 6))
    plt.plot(rolling.index, rolling["strategy"], label="Strategy")
    plt.plot(rolling.index, rolling["spy"], label="SPY")
    plt.title("Rolling 12-Month Return")
    plt.xlabel("Date")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_rolling_12m_returns.png", dpi=150)
    plt.close()

    if trades is not None and not trades.empty:
        pct_col = None
        for c in ["return_pct", "trade_return_pct", "pnl_pct", "ret_pct"]:
            if c in trades.columns:
                pct_col = c
                break
        if pct_col:
            vals = pd.to_numeric(trades[pct_col], errors="coerce").dropna()
            if vals.abs().median() < 1:
                vals = vals * 100
            plt.figure(figsize=(11, 6))
            plt.hist(vals, bins=40)
            plt.title("Trade Return Distribution")
            plt.xlabel("Trade Return (%)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("benchmark_trade_distribution.png", dpi=150)
            plt.close()


def make_html(summary: pd.DataFrame, monthly_strategy: pd.DataFrame, monthly_spy: pd.DataFrame) -> None:
    html = f"""
<html>
<head>
  <title>AlgoTrader Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; margin: 18px 0; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th {{ background: #f3f3f3; }}
    td:first-child, th:first-child {{ text-align: left; }}
    img {{ max-width: 100%; margin: 16px 0 28px; border: 1px solid #ddd; }}
    h1, h2 {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>AlgoTrader Benchmark Report</h1>
  <h2>Summary</h2>
  {summary.to_html(index=False)}

  <h2>Equity Curve</h2>
  <img src="benchmark_equity_curve.png">

  <h2>Drawdown</h2>
  <img src="benchmark_drawdown.png">

  <h2>Rolling 12-Month Returns</h2>
  <img src="benchmark_rolling_12m_returns.png">

  <h2>Strategy Monthly Returns (%)</h2>
  {monthly_strategy.to_html()}

  <h2>SPY Monthly Returns (%)</h2>
  {monthly_spy.to_html()}
</body>
</html>
"""
    with open("benchmark_report.html", "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    print("=" * 70)
    print("BENCHMARK REPORT: STRATEGY VS SPY")
    print("=" * 70)

    strategy = load_strategy_equity()
    initial = float(strategy["strategy"].iloc[0])
    spy = load_spy_equity(strategy.index[0], strategy.index[-1], initial)

    combined = strategy.join(spy, how="inner").dropna()
    if len(combined) < 30:
        raise RuntimeError("Not enough overlapping Strategy/SPY data to build benchmark report.")

    trades = pd.read_csv(TRADES_FILE) if os.path.exists(TRADES_FILE) else None

    strategy_stats = calc_stats("Strategy", combined["strategy"])
    spy_stats = calc_stats("SPY", combined["spy"])
    summary = stats_to_frame([strategy_stats, spy_stats])

    monthly_strategy = monthly_returns_table(combined["strategy"])
    monthly_spy = monthly_returns_table(combined["spy"])

    combined.to_csv("benchmark_equity_data.csv")
    summary.to_csv("benchmark_report_summary.csv", index=False)
    monthly_strategy.to_csv("benchmark_report_monthly_returns_strategy.csv")
    monthly_spy.to_csv("benchmark_report_monthly_returns_spy.csv")

    save_charts(combined, trades)
    make_html(summary, monthly_strategy, monthly_spy)

    print(summary.to_string(index=False))
    print("\nSaved:")
    print("  benchmark_report_summary.csv")
    print("  benchmark_equity_data.csv")
    print("  benchmark_report_monthly_returns_strategy.csv")
    print("  benchmark_report_monthly_returns_spy.csv")
    print("  benchmark_equity_curve.png")
    print("  benchmark_drawdown.png")
    print("  benchmark_rolling_12m_returns.png")
    print("  benchmark_trade_distribution.png, if trade returns were found")
    print("  benchmark_report.html")


if __name__ == "__main__":
    main()
