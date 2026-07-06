#!/usr/bin/env python3
"""
Compare the current heuristic screener strategy against the ML probability model.

Outputs:
  model_comparison_summary.csv
  model_comparison_equity.csv
  model_comparison_trades.csv
  model_comparison_report.html
  model_comparison_equity_curve.png
  model_comparison_drawdown.png
  model_comparison_yearly_returns.png

Run:
  python train_probability_model.py --start 2018-01-01 --model gbdt
  python run_model_comparison.py
"""

import os
import sys
import math
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


INITIAL_CAPITAL = 100_000
START_DATE = "2018-01-01"
HOLD_DAYS = 5
TOP_N = 12
MAX_POSITION_PCT = 0.15
MIN_POSITION_PCT = 0.02
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00
SLIPPAGE_BPS = 5  # 0.05% each side
PROB_THRESHOLD = 0.30

MODEL_FILE = "probability_model.pkl"
ML_SCORES_FILE = "ml_probability_predictions.csv"
TRAINING_DATA_FILE = "probability_training_dataset.csv"


def commission(shares: int) -> float:
    return max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE)


def pct_change(a, b):
    if a == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return b / a - 1


def max_drawdown(equity):
    s = pd.Series(equity).astype(float)
    peak = s.cummax()
    dd = s / peak - 1
    return float(dd.min()), dd


def annualized_return(equity, dates):
    if len(equity) < 2:
        return np.nan
    years = (pd.to_datetime(dates.iloc[-1]) - pd.to_datetime(dates.iloc[0])).days / 365.25
    if years <= 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def sharpe_ratio(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return np.sqrt(252) * r.mean() / r.std()


def sortino_ratio(returns):
    r = pd.Series(returns).dropna()
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return np.nan
    return np.sqrt(252) * r.mean() / downside.std()


def performance_metrics(equity_df, name):
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    eq = df["equity"].astype(float)
    rets = eq.pct_change().dropna()
    mdd, _ = max_drawdown(eq)
    cagr = annualized_return(eq, df["date"])
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else np.nan
    sharpe = sharpe_ratio(rets)
    sortino = sortino_ratio(rets)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    return {
        "strategy": name,
        "start_date": df["date"].iloc[0].date(),
        "end_date": df["date"].iloc[-1].date(),
        "initial_equity": eq.iloc[0],
        "ending_equity": eq.iloc[-1],
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "annual_vol_pct": vol * 100 if not pd.isna(vol) else np.nan,
        "max_drawdown_pct": mdd * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
    }


def load_ml_predictions():
    if not os.path.exists(ML_SCORES_FILE):
        raise FileNotFoundError(
            f"{ML_SCORES_FILE} not found. Run: python train_probability_model.py --start 2018-01-01 --model gbdt"
        )

    df = pd.read_csv(ML_SCORES_FILE)
    df["date"] = pd.to_datetime(df["date"])

    prob_col = None
    for c in ["ml_prob_up_5pct", "prob_up_5pct", "probability", "pred_prob"]:
        if c in df.columns:
            prob_col = c
            break
    if prob_col is None:
        raise ValueError(f"No ML probability column found in {ML_SCORES_FILE}")

    df["ml_prob"] = pd.to_numeric(df[prob_col], errors="coerce")
    if df["ml_prob"].max() > 1.5:
        df["ml_prob"] = df["ml_prob"] / 100.0

    return df.dropna(subset=["ticker", "date", "ml_prob"])


def load_training_dataset():
    if not os.path.exists(TRAINING_DATA_FILE):
        raise FileNotFoundError(
            f"{TRAINING_DATA_FILE} not found. Run: python train_probability_model.py --start 2018-01-01 --model gbdt"
        )
    df = pd.read_csv(TRAINING_DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_heuristic_scores(data):
    """
    Rebuild a simple heuristic score from the training dataset features.
    This mirrors the old idea: volatility-adjusted probability plus signal bonuses.
    """
    df = data.copy()

    # Base hit rate by volatility bucket approximation
    base = df.get("target_up_5pct", pd.Series(index=df.index, data=0)).mean()
    if pd.isna(base) or base <= 0:
        base = 0.12

    score = pd.Series(base, index=df.index, dtype=float)

    # Momentum and trend signals
    if "ret_21d" in df.columns:
        score += np.where(df["ret_21d"] > df["ret_21d"].quantile(0.70), 0.05, 0)
    if "ret_63d" in df.columns:
        score += np.where(df["ret_63d"] > df["ret_63d"].quantile(0.70), 0.05, 0)
    if "dist_ma_20d" in df.columns:
        score += np.where(df["dist_ma_20d"] > 0, 0.03, 0)
    if "dist_ma_50d" in df.columns:
        score += np.where(df["dist_ma_50d"] > 0, 0.03, 0)
    if "dist_52w_high" in df.columns:
        score += np.where(df["dist_52w_high"] > -0.15, 0.04, 0)

    # Mean reversion / not too overextended
    if "rsi_14d" in df.columns:
        score += np.where((df["rsi_14d"] >= 45) & (df["rsi_14d"] <= 75), 0.03, 0)

    # Market regime
    if "spy_above_200d" in df.columns:
        score += np.where(df["spy_above_200d"] > 0, 0.04, -0.04)

    # Volatility sweet spot
    if "vol_21d" in df.columns:
        score += np.where((df["vol_21d"] >= 0.20) & (df["vol_21d"] <= 0.80), 0.04, 0)

    df["heuristic_prob"] = score.clip(0.01, 0.80)
    return df


def simulate_strategy(scores, score_col, name):
    """
    Weekly rebalance simulation:
    - Rank stocks by score.
    - Buy top names above threshold.
    - Equal-ish max-capped allocation.
    - Hold 5 trading days.
    - Include commissions and slippage.
    """
    df = scores.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", score_col], ascending=[True, False])

    required_cols = {"date", "ticker", score_col, "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {name}: {missing}")

    if "future_5d_return" not in df.columns:
        # Try common names
        for c in ["ret_fwd_5d", "forward_5d_return", "target_return_5d"]:
            if c in df.columns:
                df["future_5d_return"] = df[c]
                break
    if "future_5d_return" not in df.columns:
        # Estimate from close and future close if available
        if "future_close_5d" in df.columns:
            df["future_5d_return"] = df["future_close_5d"] / df["close"] - 1
        else:
            raise ValueError("No future 5-day return column found in training/prediction data.")

    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 20:
        raise ValueError("Not enough dates to simulate.")

    # Use every HOLD_DAYS-th date as rebalance date
    rebalance_dates = dates[::HOLD_DAYS]

    cash = INITIAL_CAPITAL
    equity_records = []
    trades = []

    for i, dt in enumerate(rebalance_dates):
        day = df[df["date"] == dt].copy()
        day = day.dropna(subset=[score_col, "close", "future_5d_return"])
        day = day[day[score_col] >= PROB_THRESHOLD].sort_values(score_col, ascending=False).head(TOP_N)

        starting_equity = cash
        if day.empty:
            equity_records.append({"date": dt, "equity": cash, "strategy": name})
            continue

        selected = []
        running_cost = 0.0

        for _, row in day.iterrows():
            remaining = cash - running_cost
            if remaining <= 0:
                break

            target_cost = min(cash * MAX_POSITION_PCT, remaining)
            if target_cost < cash * MIN_POSITION_PCT:
                continue

            entry_price = float(row["close"]) * (1 + SLIPPAGE_BPS / 10000)
            shares = int((target_cost - MIN_COMMISSION) / entry_price)
            if shares <= 0:
                continue

            while shares > 0:
                buy_commission = commission(shares)
                trade_value = shares * entry_price
                total_entry_cost = trade_value + buy_commission

                if total_entry_cost <= remaining and total_entry_cost >= cash * MIN_POSITION_PCT:
                    break
                shares -= 1

            if shares <= 0:
                continue

            fwd_ret = float(row["future_5d_return"])
            exit_price = entry_price * (1 + fwd_ret) * (1 - SLIPPAGE_BPS / 10000)
            sell_commission = commission(shares)
            exit_value = shares * exit_price - sell_commission
            pnl = exit_value - total_entry_cost

            running_cost += total_entry_cost
            selected.append(pnl)

            trades.append({
                "strategy": name,
                "entry_date": dt,
                "ticker": row["ticker"],
                "score": row[score_col],
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_cost": total_entry_cost,
                "exit_value": exit_value,
                "commission": buy_commission + sell_commission,
                "gross_return_pct": fwd_ret * 100,
                "net_pnl": pnl,
                "net_return_pct": pnl / total_entry_cost * 100 if total_entry_cost else np.nan,
            })

        cash = cash + sum(selected)
        equity_records.append({"date": dt, "equity": cash, "strategy": name})

    return pd.DataFrame(equity_records), pd.DataFrame(trades)


def get_spy_benchmark(start, end):
    if yf is None:
        return pd.DataFrame()

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
    spy["equity"] = INITIAL_CAPITAL * spy["close"] / spy["close"].iloc[0]
    spy["strategy"] = "SPY"
    return spy[["date", "equity", "strategy"]]


def yearly_returns(equity):
    df = equity.copy()
    df["date"] = pd.to_datetime(df["date"])
    out = []
    for strat, g in df.groupby("strategy"):
        g = g.sort_values("date")
        g["year"] = g["date"].dt.year
        for y, yg in g.groupby("year"):
            ret = yg["equity"].iloc[-1] / yg["equity"].iloc[0] - 1
            out.append({"strategy": strat, "year": y, "return_pct": ret * 100})
    return pd.DataFrame(out)


def make_charts(equity_all, summary, trades):
    if plt is None:
        return []

    files = []

    # Equity curve
    plt.figure(figsize=(12, 6))
    for strat, g in equity_all.groupby("strategy"):
        g = g.sort_values("date")
        plt.plot(g["date"], g["equity"], label=strat)
    plt.title("Equity Curve: Heuristic vs ML vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    f = "model_comparison_equity_curve.png"
    plt.savefig(f, dpi=160)
    plt.close()
    files.append(f)

    # Drawdown
    plt.figure(figsize=(12, 6))
    for strat, g in equity_all.groupby("strategy"):
        g = g.sort_values("date")
        _, dd = max_drawdown(g["equity"])
        plt.plot(g["date"], dd * 100, label=strat)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    f = "model_comparison_drawdown.png"
    plt.savefig(f, dpi=160)
    plt.close()
    files.append(f)

    # Yearly returns
    yr = yearly_returns(equity_all)
    if not yr.empty:
        pivot = yr.pivot(index="year", columns="strategy", values="return_pct")
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Yearly Returns")
        ax.set_ylabel("Return (%)")
        plt.tight_layout()
        f = "model_comparison_yearly_returns.png"
        plt.savefig(f, dpi=160)
        plt.close()
        files.append(f)

    # Trade return distribution
    if not trades.empty:
        plt.figure(figsize=(12, 6))
        for strat, g in trades.groupby("strategy"):
            plt.hist(g["net_return_pct"].dropna(), bins=60, alpha=0.5, label=strat)
        plt.title("Trade Return Distribution")
        plt.xlabel("Net trade return (%)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        f = "model_comparison_trade_distribution.png"
        plt.savefig(f, dpi=160)
        plt.close()
        files.append(f)

    return files


def make_html_report(summary, yearly, trades_summary, chart_files):
    html = []
    html.append("<html><head><title>Model Comparison Report</title>")
    html.append("<style>body{font-family:Arial;margin:40px;} table{border-collapse:collapse;margin:20px 0;} th,td{border:1px solid #ddd;padding:8px;text-align:right;} th{text-align:center;background:#f2f2f2;} h1,h2{color:#222;} img{max-width:100%;margin:20px 0;}</style>")
    html.append("</head><body>")
    html.append("<h1>Model Comparison Report</h1>")
    html.append("<p>Compares the current heuristic probability strategy against the ML probability model and SPY.</p>")
    html.append(f"<p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append("<h2>Overall Summary</h2>")
    html.append(summary.to_html(index=False, float_format=lambda x: f"{x:,.2f}"))
    html.append("<h2>Yearly Returns</h2>")
    html.append(yearly.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not yearly.empty else "<p>No yearly data.</p>")
    html.append("<h2>Trade Summary</h2>")
    html.append(trades_summary.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not trades_summary.empty else "<p>No trade data.</p>")
    html.append("<h2>Charts</h2>")
    for f in chart_files:
        html.append(f"<h3>{f}</h3><img src='{f}' />")
    html.append("</body></html>")

    with open("model_comparison_report.html", "w", encoding="utf-8") as out:
        out.write("\n".join(html))


def main():
    print("=" * 70)
    print("MODEL COMPARISON: HEURISTIC VS ML PROBABILITY MODEL")
    print("=" * 70)

    ml = load_ml_predictions()
    data = load_training_dataset()
    heuristic = build_heuristic_scores(data)

    # Ensure future return exists in ML; join from training dataset if needed
    if "future_5d_return" not in ml.columns:
        join_cols = ["date", "ticker"]
        fut_col = None
        for c in ["future_5d_return", "ret_fwd_5d", "forward_5d_return", "target_return_5d"]:
            if c in data.columns:
                fut_col = c
                break
        if fut_col is not None:
            ml = ml.merge(data[join_cols + [fut_col]], on=join_cols, how="left")
            ml["future_5d_return"] = ml[fut_col]
        elif "future_close_5d" in data.columns and "close" in data.columns:
            tmp = data[join_cols + ["close", "future_close_5d"]].copy()
            tmp["future_5d_return"] = tmp["future_close_5d"] / tmp["close"] - 1
            ml = ml.merge(tmp[join_cols + ["future_5d_return"]], on=join_cols, how="left")

    print("Simulating heuristic strategy...")
    heuristic_equity, heuristic_trades = simulate_strategy(heuristic, "heuristic_prob", "Heuristic")

    print("Simulating ML strategy...")
    ml_equity, ml_trades = simulate_strategy(ml, "ml_prob", "ML Probability")

    start = min(heuristic_equity["date"].min(), ml_equity["date"].min())
    end = max(heuristic_equity["date"].max(), ml_equity["date"].max())
    print("Downloading SPY benchmark...")
    spy_equity = get_spy_benchmark(start, end)

    equity_all = pd.concat([heuristic_equity, ml_equity, spy_equity], ignore_index=True)
    trades_all = pd.concat([heuristic_trades, ml_trades], ignore_index=True)

    summary_rows = []
    for strat, g in equity_all.groupby("strategy"):
        summary_rows.append(performance_metrics(g, strat))
    summary = pd.DataFrame(summary_rows)

    trade_summary = []
    for strat, g in trades_all.groupby("strategy"):
        wins = g[g["net_pnl"] > 0]
        losses = g[g["net_pnl"] <= 0]
        trade_summary.append({
            "strategy": strat,
            "trades": len(g),
            "win_rate_pct": (g["net_pnl"] > 0).mean() * 100,
            "avg_trade_pct": g["net_return_pct"].mean(),
            "avg_win_pct": wins["net_return_pct"].mean() if len(wins) else np.nan,
            "avg_loss_pct": losses["net_return_pct"].mean() if len(losses) else np.nan,
            "total_commission": g["commission"].sum(),
            "profit_factor": wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()) if len(losses) and losses["net_pnl"].sum() != 0 else np.nan,
        })
    trade_summary = pd.DataFrame(trade_summary)

    yearly = yearly_returns(equity_all)

    chart_files = make_charts(equity_all, summary, trades_all)
    make_html_report(summary, yearly, trade_summary, chart_files)

    summary.to_csv("model_comparison_summary.csv", index=False)
    equity_all.to_csv("model_comparison_equity.csv", index=False)
    trades_all.to_csv("model_comparison_trades.csv", index=False)
    yearly.to_csv("model_comparison_yearly_returns.csv", index=False)
    trade_summary.to_csv("model_comparison_trade_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nTrade summary:")
    print(trade_summary.to_string(index=False))
    print("\nSaved:")
    print("  model_comparison_summary.csv")
    print("  model_comparison_equity.csv")
    print("  model_comparison_trades.csv")
    print("  model_comparison_yearly_returns.csv")
    print("  model_comparison_trade_summary.csv")
    print("  model_comparison_report.html")
    for f in chart_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
