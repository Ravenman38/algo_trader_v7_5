#!/usr/bin/env python3
"""
AlgoTrader v6 Validation Suite

Purpose:
  Validate whether the ML strategy results are robust enough to consider
  paper trading.

Run after V5 walk-forward comparison:

  python run.py train-ml --start 2018-01-01 --model gbdt
  python run.py walkforward
  python run_validation_suite.py

Inputs expected:
  walkforward_comparison_overall.csv
  walkforward_comparison_yearly.csv
  walkforward_comparison_regimes.csv
  walkforward_comparison_equity.csv
  walkforward_comparison_trades.csv
  probability_model_summary.csv
  probability_model_feature_importance.csv

Outputs:
  validation_report.html
  validation_summary.csv
  validation_bootstrap.csv
  validation_trade_concentration.csv
  validation_monthly_returns.csv
  validation_equity_rolling_metrics.csv
  validation_trade_distribution.png
  validation_monthly_returns.png
  validation_rolling_sharpe.png
  validation_rolling_drawdown.png
"""

import os
import math
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BOOTSTRAP_ITERATIONS = 2000
RANDOM_SEED = 42
TRADING_DAYS = 252


REQUIRED_FILES = [
    "walkforward_comparison_overall.csv",
    "walkforward_comparison_yearly.csv",
    "walkforward_comparison_regimes.csv",
    "walkforward_comparison_equity.csv",
    "walkforward_comparison_trades.csv",
    "probability_model_summary.csv",
    "probability_model_feature_importance.csv",
]


def require_files():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nRun first:\n"
            + "  python run.py train-ml --start 2018-01-01 --model gbdt\n"
            + "  python run.py walkforward"
        )


def load_inputs():
    require_files()

    overall = pd.read_csv("walkforward_comparison_overall.csv")
    yearly = pd.read_csv("walkforward_comparison_yearly.csv")
    regimes = pd.read_csv("walkforward_comparison_regimes.csv")
    equity = pd.read_csv("walkforward_comparison_equity.csv")
    trades = pd.read_csv("walkforward_comparison_trades.csv")
    model_summary = pd.read_csv("probability_model_summary.csv")
    feature_importance = pd.read_csv("probability_model_feature_importance.csv")

    equity["date"] = pd.to_datetime(equity["date"])
    if "date" in trades.columns:
        trades["date"] = pd.to_datetime(trades["date"])
    elif "entry_date" in trades.columns:
        trades["date"] = pd.to_datetime(trades["entry_date"])

    return overall, yearly, regimes, equity, trades, model_summary, feature_importance


def max_drawdown(series):
    s = pd.Series(series).astype(float)
    peak = s.cummax()
    dd = s / peak - 1
    return dd.min(), dd


def sharpe_ratio(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return np.sqrt(TRADING_DAYS) * r.mean() / r.std()


def cagr(equity, dates):
    e = pd.Series(equity).astype(float)
    d = pd.to_datetime(dates)
    if len(e) < 2:
        return np.nan
    years = (d.iloc[-1] - d.iloc[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1


def get_ml_trades(trades):
    asset_col = "asset" if "asset" in trades.columns else "strategy"
    if asset_col not in trades.columns:
        return trades.copy()

    ml = trades[trades[asset_col].astype(str).str.contains("ML", case=False, na=False)].copy()
    if ml.empty:
        ml = trades.copy()
    return ml


def get_ml_equity(equity):
    asset_col = "asset" if "asset" in equity.columns else "strategy"
    ml = equity[equity[asset_col].astype(str).str.contains("ML", case=False, na=False)].copy()
    if ml.empty:
        ml = equity.copy()
    return ml.sort_values("date")


def bootstrap_trade_confidence(trades, iterations=BOOTSTRAP_ITERATIONS):
    np.random.seed(RANDOM_SEED)

    ml = get_ml_trades(trades)
    if ml.empty or "net_return_pct" not in ml.columns:
        return pd.DataFrame()

    returns = ml["net_return_pct"].dropna().values / 100.0
    if len(returns) < 20:
        return pd.DataFrame()

    rows = []
    n = len(returns)

    for _ in range(iterations):
        sample = np.random.choice(returns, size=n, replace=True)
        total_return = np.prod(1 + sample) - 1

        # Approximate trade-level Sharpe, not daily Sharpe.
        sample_sharpe = sample.mean() / sample.std() if sample.std() > 0 else np.nan
        win_rate = (sample > 0).mean()
        avg_trade = sample.mean()

        gross_win = sample[sample > 0].sum()
        gross_loss = abs(sample[sample <= 0].sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else np.nan

        rows.append({
            "total_return_pct": total_return * 100,
            "trade_sharpe": sample_sharpe,
            "win_rate_pct": win_rate * 100,
            "avg_trade_pct": avg_trade * 100,
            "profit_factor": profit_factor,
        })

    boot = pd.DataFrame(rows)

    summary = []
    for col in boot.columns:
        summary.append({
            "metric": col,
            "mean": boot[col].mean(),
            "p05": boot[col].quantile(0.05),
            "p50": boot[col].quantile(0.50),
            "p95": boot[col].quantile(0.95),
        })

    out = pd.DataFrame(summary)
    out.to_csv("validation_bootstrap.csv", index=False)
    return out


def trade_concentration(trades):
    ml = get_ml_trades(trades)
    if ml.empty:
        return pd.DataFrame()

    if "net_pnl" not in ml.columns:
        return pd.DataFrame()

    by_ticker = (
        ml.groupby("ticker", dropna=False)
        .agg(
            trades=("ticker", "count"),
            total_pnl=("net_pnl", "sum"),
            avg_return_pct=("net_return_pct", "mean") if "net_return_pct" in ml.columns else ("net_pnl", "mean"),
            win_rate_pct=("net_pnl", lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
        .sort_values("total_pnl", ascending=False)
    )

    total_positive = by_ticker[by_ticker["total_pnl"] > 0]["total_pnl"].sum()
    if total_positive != 0:
        by_ticker["pct_of_positive_pnl"] = np.where(
            by_ticker["total_pnl"] > 0,
            by_ticker["total_pnl"] / total_positive * 100,
            0,
        )
    else:
        by_ticker["pct_of_positive_pnl"] = np.nan

    by_ticker.to_csv("validation_trade_concentration.csv", index=False)
    return by_ticker


def monthly_returns(equity):
    ml = get_ml_equity(equity)
    if ml.empty:
        return pd.DataFrame()

    ml = ml.sort_values("date").copy()
    ml["month"] = ml["date"].dt.to_period("M")

    rows = []
    for month, g in ml.groupby("month"):
        ret = g["equity"].iloc[-1] / g["equity"].iloc[0] - 1
        rows.append({
            "month": str(month),
            "year": month.year,
            "month_num": month.month,
            "return_pct": ret * 100,
        })

    out = pd.DataFrame(rows)
    out.to_csv("validation_monthly_returns.csv", index=False)
    return out


def rolling_metrics(equity):
    ml = get_ml_equity(equity)
    if ml.empty:
        return pd.DataFrame()

    ml = ml.sort_values("date").copy()
    ml["return"] = ml["equity"].pct_change()
    ml["rolling_sharpe_63d"] = ml["return"].rolling(63).apply(sharpe_ratio, raw=False)

    dd_min, dd_series = max_drawdown(ml["equity"])
    ml["drawdown_pct"] = dd_series.values * 100
    ml["rolling_max_drawdown_63d_pct"] = (
        ml["equity"]
        .rolling(63)
        .apply(lambda x: max_drawdown(pd.Series(x))[0] * 100, raw=False)
    )

    out = ml[["date", "equity", "return", "rolling_sharpe_63d", "drawdown_pct", "rolling_max_drawdown_63d_pct"]]
    out.to_csv("validation_equity_rolling_metrics.csv", index=False)
    return out


def validation_checks(overall, yearly, regimes, equity, trades, model_summary, feature_importance):
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    # ML vs SPY
    ml_row = overall[overall["asset"].astype(str).str.contains("ML Strategy", case=False, na=False)]
    spy_row = overall[overall["asset"].astype(str).eq("SPY")]

    if not ml_row.empty and not spy_row.empty:
        ml = ml_row.iloc[0]
        spy = spy_row.iloc[0]

        add(
            "ML CAGR beats SPY",
            "PASS" if ml["cagr_pct"] > spy["cagr_pct"] else "FAIL",
            f"ML {ml['cagr_pct']:.2f}% vs SPY {spy['cagr_pct']:.2f}%",
        )

        add(
            "ML Sharpe beats SPY",
            "PASS" if ml["sharpe"] > spy["sharpe"] else "FAIL",
            f"ML {ml['sharpe']:.2f} vs SPY {spy['sharpe']:.2f}",
        )

        add(
            "ML drawdown not worse than SPY by more than 10 pct points",
            "PASS" if ml["max_drawdown_pct"] >= spy["max_drawdown_pct"] - 10 else "WARN",
            f"ML {ml['max_drawdown_pct']:.2f}% vs SPY {spy['max_drawdown_pct']:.2f}%",
        )

    # Yearly consistency
    if "ml_beat_spy" in yearly.columns:
        beat_count = yearly["ml_beat_spy"].fillna(False).sum()
        total_years = yearly["ml_beat_spy"].notna().sum()
        add(
            "ML beats SPY in majority of years",
            "PASS" if beat_count > total_years / 2 else "FAIL",
            f"{int(beat_count)} of {int(total_years)} years",
        )

    # Regime consistency
    if "ml_beat_spy" in regimes.columns:
        beat_count = regimes["ml_beat_spy"].fillna(False).sum()
        total = regimes["ml_beat_spy"].notna().sum()
        add(
            "ML beats SPY in majority of regimes",
            "PASS" if beat_count > total / 2 else "WARN",
            f"{int(beat_count)} of {int(total)} regimes",
        )

    # Model quality
    overall_model = model_summary[model_summary["test_year"].astype(str).str.upper().eq("OVERALL")]
    if not overall_model.empty:
        auc = float(overall_model.iloc[0]["auc"])
        edge = float(overall_model.iloc[0]["hit_rate_edge"])
        add("Model AUC above 0.60", "PASS" if auc >= 0.60 else "FAIL", f"AUC {auc:.3f}")
        add("Top-bottom hit-rate edge positive", "PASS" if edge > 0 else "FAIL", f"Edge {edge:.3f}")

    # Feature concentration
    if not feature_importance.empty and "importance" in feature_importance.columns:
        total_imp = feature_importance["importance"].sum()
        top1 = feature_importance["importance"].iloc[0] / total_imp if total_imp else np.nan
        top5 = feature_importance["importance"].iloc[:5].sum() / total_imp if total_imp else np.nan

        add(
            "Feature importance not dominated by one feature",
            "PASS" if top1 < 0.25 else "WARN",
            f"Top feature share {top1:.1%}",
        )
        add(
            "Top 5 feature concentration",
            "PASS" if top5 < 0.70 else "WARN",
            f"Top 5 share {top5:.1%}",
        )

    # Trade count
    ml_trades = get_ml_trades(trades)
    add(
        "Sufficient ML trade count",
        "PASS" if len(ml_trades) >= 300 else "WARN",
        f"{len(ml_trades)} trades",
    )

    if len(ml_trades) and "net_pnl" in ml_trades.columns:
        win_rate = (ml_trades["net_pnl"] > 0).mean()
        add(
            "ML win rate above 50%",
            "PASS" if win_rate > 0.50 else "WARN",
            f"Win rate {win_rate:.1%}",
        )

    return pd.DataFrame(checks)


def make_charts(equity, trades, monthly, rolling):
    chart_files = []

    ml_trades = get_ml_trades(trades)

    if not ml_trades.empty and "net_return_pct" in ml_trades.columns:
        plt.figure(figsize=(12, 6))
        plt.hist(ml_trades["net_return_pct"].dropna(), bins=60)
        plt.title("ML Trade Return Distribution")
        plt.xlabel("Net trade return (%)")
        plt.ylabel("Count")
        plt.tight_layout()
        f = "validation_trade_distribution.png"
        plt.savefig(f, dpi=160)
        plt.close()
        chart_files.append(f)

    if not monthly.empty:
        plt.figure(figsize=(12, 6))
        plt.bar(monthly["month"], monthly["return_pct"])
        plt.xticks(rotation=90)
        plt.title("ML Monthly Returns")
        plt.ylabel("Return (%)")
        plt.tight_layout()
        f = "validation_monthly_returns.png"
        plt.savefig(f, dpi=160)
        plt.close()
        chart_files.append(f)

    if not rolling.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(rolling["date"]), rolling["rolling_sharpe_63d"])
        plt.axhline(0, linewidth=1)
        plt.title("ML Rolling 63-Period Sharpe")
        plt.xlabel("Date")
        plt.ylabel("Rolling Sharpe")
        plt.tight_layout()
        f = "validation_rolling_sharpe.png"
        plt.savefig(f, dpi=160)
        plt.close()
        chart_files.append(f)

        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(rolling["date"]), rolling["drawdown_pct"])
        plt.title("ML Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.tight_layout()
        f = "validation_rolling_drawdown.png"
        plt.savefig(f, dpi=160)
        plt.close()
        chart_files.append(f)

    return chart_files


def generate_html(checks, overall, yearly, regimes, bootstrap, concentration, monthly, chart_files):
    pass_count = (checks["status"] == "PASS").sum()
    warn_count = (checks["status"] == "WARN").sum()
    fail_count = (checks["status"] == "FAIL").sum()

    if fail_count > 0:
        verdict = "NOT READY"
        verdict_color = "#ffebee"
        border_color = "#c62828"
    elif warn_count >= 3:
        verdict = "PROMISING BUT NEEDS REVIEW"
        verdict_color = "#fff8dc"
        border_color = "#e0a800"
    else:
        verdict = "PROMISING - READY FOR PAPER TRADING VALIDATION"
        verdict_color = "#eef7ee"
        border_color = "#2e7d32"

    html = []
    html.append("<html><head><title>AlgoTrader v6 Validation Report</title>")
    html.append("""
<style>
body { font-family: Arial, sans-serif; margin: 40px; color: #222; }
table { border-collapse: collapse; margin: 20px 0; width: 100%; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 7px; text-align: right; }
th { background: #f3f3f3; text-align: center; }
td:first-child, th:first-child { text-align: left; }
h1, h2 { color: #111; }
img { max-width: 100%; margin: 16px 0 32px 0; border: 1px solid #ddd; }
.verdict { padding: 16px; margin: 20px 0; font-size: 18px; font-weight: bold; }
.note { background: #fff8dc; padding: 12px; border-left: 4px solid #e0b000; margin: 20px 0; }
.pass { color: #2e7d32; font-weight: bold; }
.warn { color: #e0a800; font-weight: bold; }
.fail { color: #c62828; font-weight: bold; }
</style>
""")
    html.append("</head><body>")
    html.append("<h1>AlgoTrader v6 Validation Report</h1>")
    html.append(f"<p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append("<div class='note'>This report tries to falsify the strategy. Passing these checks does not guarantee future performance, but it increases confidence that the backtest is not obviously fragile.</div>")
    html.append(f"<div class='verdict' style='background:{verdict_color}; border-left:5px solid {border_color};'>Overall verdict: {verdict}<br>Checks: {pass_count} pass, {warn_count} warn, {fail_count} fail</div>")

    html.append("<h2>Validation Checks</h2>")
    html.append(checks.to_html(index=False, escape=False))

    html.append("<h2>Overall Walk-Forward Performance</h2>")
    html.append(overall.to_html(index=False, float_format=lambda x: f"{x:,.2f}"))

    html.append("<h2>Year-by-Year Performance</h2>")
    html.append(yearly.to_html(index=False, float_format=lambda x: f"{x:,.2f}"))

    html.append("<h2>Market Regime Performance</h2>")
    html.append(regimes.to_html(index=False, float_format=lambda x: f"{x:,.2f}"))

    html.append("<h2>Bootstrap Trade Confidence</h2>")
    if bootstrap.empty:
        html.append("<p>Bootstrap unavailable. Not enough trade-return data.</p>")
    else:
        html.append(bootstrap.to_html(index=False, float_format=lambda x: f"{x:,.2f}"))

    html.append("<h2>Top Trade Contributors</h2>")
    if concentration.empty:
        html.append("<p>Concentration data unavailable.</p>")
    else:
        html.append(concentration.head(25).to_html(index=False, float_format=lambda x: f"{x:,.2f}"))

    html.append("<h2>Monthly Returns</h2>")
    if monthly.empty:
        html.append("<p>Monthly data unavailable.</p>")
    else:
        html.append(monthly.tail(36).to_html(index=False, float_format=lambda x: f"{x:,.2f}"))

    html.append("<h2>Charts</h2>")
    for f in chart_files:
        html.append(f"<h3>{f}</h3>")
        html.append(f"<img src='{f}' />")

    html.append("</body></html>")

    with open("validation_report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def main():
    print("=" * 78)
    print("ALGOTRADER V6 VALIDATION SUITE")
    print("=" * 78)

    overall, yearly, regimes, equity, trades, model_summary, feature_importance = load_inputs()

    print("Running validation checks...")
    checks = validation_checks(overall, yearly, regimes, equity, trades, model_summary, feature_importance)

    print("Running bootstrap confidence analysis...")
    bootstrap = bootstrap_trade_confidence(trades)

    print("Analyzing trade concentration...")
    concentration = trade_concentration(trades)

    print("Calculating monthly returns...")
    monthly = monthly_returns(equity)

    print("Calculating rolling metrics...")
    rolling = rolling_metrics(equity)

    print("Generating charts...")
    chart_files = make_charts(equity, trades, monthly, rolling)

    print("Writing validation report...")
    generate_html(checks, overall, yearly, regimes, bootstrap, concentration, monthly, chart_files)

    checks.to_csv("validation_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("VALIDATION CHECKS")
    print("=" * 78)
    print(checks.to_string(index=False))

    print("\nSaved:")
    print("  validation_report.html")
    print("  validation_summary.csv")
    print("  validation_bootstrap.csv")
    print("  validation_trade_concentration.csv")
    print("  validation_monthly_returns.csv")
    print("  validation_equity_rolling_metrics.csv")
    for f in chart_files:
        print(f"  {f}")

    print("\nOpen validation_report.html for the full report.")


if __name__ == "__main__":
    main()
