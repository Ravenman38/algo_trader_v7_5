#!/usr/bin/env python3
"""
ML PERIOD ANALYSIS: ML strategy vs Heuristic strategy vs SPY

Purpose:
  Runs a 2018+ comparison using the trained ML probability model outputs.

Required first:
  python train_probability_model.py --start 2018-01-01 --model gbdt

Then run:
  python run_ml_period_analysis.py

Outputs:
  ml_period_analysis_overall.csv
  ml_period_analysis_yearly.csv
  ml_period_analysis_regimes.csv
  ml_period_analysis_equity.csv
  ml_period_analysis_trades.csv
  ml_period_analysis_report.html
  ml_period_analysis_equity_curve.png
  ml_period_analysis_drawdown.png
  ml_period_analysis_yearly_returns.png
"""

import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


INITIAL_CAPITAL = 100_000
START_DATE = "2018-01-01"
HOLD_DAYS = 5

TOP_N = 12
MAX_POSITION_PCT = 0.15
MIN_POSITION_PCT = 0.02

COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00
SLIPPAGE_BPS = 5

ML_PREDICTIONS_FILE = "ml_probability_predictions.csv"
TRAINING_DATA_FILE = "probability_training_dataset.csv"

PERIODS = [
    ("2018 correction", "2018-01-01", "2018-12-31"),
    ("2019 bull market", "2019-01-01", "2019-12-31"),
    ("2020 COVID crash", "2020-02-19", "2020-03-23"),
    ("2020-2021 recovery", "2020-03-24", "2021-12-31"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
    ("2023-2024 bull / AI rally", "2023-01-01", "2024-12-31"),
    ("2025-present", "2025-01-01", None),
]


def commission(shares: int) -> float:
    return max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE)


def max_drawdown(equity):
    s = pd.Series(equity).astype(float)
    peak = s.cummax()
    dd = s / peak - 1
    return float(dd.min()), dd


def sharpe_ratio(rets):
    r = pd.Series(rets).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return np.sqrt(252) * r.mean() / r.std()


def sortino_ratio(rets):
    r = pd.Series(rets).dropna()
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return np.nan
    return np.sqrt(252) * r.mean() / downside.std()


def cagr(equity, dates):
    eq = pd.Series(equity).astype(float)
    dt = pd.to_datetime(dates)
    if len(eq) < 2:
        return np.nan
    years = (dt.iloc[-1] - dt.iloc[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1


def get_return_col(df):
    for c in ["future_5d_return", "forward_return_5d", "ret_fwd_5d", "forward_5d_return", "target_return_5d"]:
        if c in df.columns:
            return c
    if "future_close_5d" in df.columns and "close" in df.columns:
        df["future_5d_return"] = df["future_close_5d"] / df["close"] - 1
        return "future_5d_return"
    raise ValueError("Could not find future 5-day return column.")


def load_data():
    if not os.path.exists(ML_PREDICTIONS_FILE):
        raise FileNotFoundError(
            f"{ML_PREDICTIONS_FILE} not found. Run: "
            "python train_probability_model.py --start 2018-01-01 --model gbdt"
        )

    if not os.path.exists(TRAINING_DATA_FILE):
        raise FileNotFoundError(
            f"{TRAINING_DATA_FILE} not found. Run: "
            "python train_probability_model.py --start 2018-01-01 --model gbdt"
        )

    ml = pd.read_csv(ML_PREDICTIONS_FILE)
    data = pd.read_csv(TRAINING_DATA_FILE)

    ml["date"] = pd.to_datetime(ml["date"])
    data["date"] = pd.to_datetime(data["date"])

    # Normalize ML probability column.
    prob_col = None
    for c in ["ml_prob_up_5pct", "ml_prob", "prob_up_5pct", "pred_prob", "probability"]:
        if c in ml.columns:
            prob_col = c
            break

    if prob_col is None:
        raise ValueError("Could not find ML probability column in ml_probability_predictions.csv")

    ml["ml_score"] = pd.to_numeric(ml[prob_col], errors="coerce")
    if ml["ml_score"].max() > 1.5:
        ml["ml_score"] = ml["ml_score"] / 100.0

    # Attach forward returns if missing from ML predictions.
    ret_col_data = get_return_col(data)
    if "future_5d_return" not in ml.columns:
        ml = ml.merge(
            data[["date", "ticker", ret_col_data]].rename(columns={ret_col_data: "future_5d_return"}),
            on=["date", "ticker"],
            how="left",
        )
    else:
        ml["future_5d_return"] = pd.to_numeric(ml["future_5d_return"], errors="coerce")

    if "close" not in ml.columns:
        ml = ml.merge(data[["date", "ticker", "close"]], on=["date", "ticker"], how="left")

    heuristic = build_heuristic_scores(data)
    return ml, heuristic


def build_heuristic_scores(df):
    out = df.copy()
    if "target_up_5pct" in out.columns:
        base = float(out["target_up_5pct"].mean())
    else:
        base = 0.12

    score = pd.Series(base, index=out.index, dtype=float)

    if "ret_21d" in out.columns:
        score += np.where(out["ret_21d"] > out["ret_21d"].quantile(0.70), 0.05, 0)
    if "ret_63d" in out.columns:
        score += np.where(out["ret_63d"] > out["ret_63d"].quantile(0.70), 0.05, 0)
    if "dist_ma_20d" in out.columns:
        score += np.where(out["dist_ma_20d"] > 0, 0.03, 0)
    if "dist_ma_50d" in out.columns:
        score += np.where(out["dist_ma_50d"] > 0, 0.03, 0)
    if "dist_52w_high" in out.columns:
        score += np.where(out["dist_52w_high"] > -0.15, 0.04, 0)
    if "rsi_14d" in out.columns:
        score += np.where((out["rsi_14d"] >= 45) & (out["rsi_14d"] <= 75), 0.03, 0)
    if "spy_above_200d" in out.columns:
        score += np.where(out["spy_above_200d"] > 0, 0.04, -0.04)
    if "vol_21d" in out.columns:
        score += np.where((out["vol_21d"] >= 0.20) & (out["vol_21d"] <= 0.80), 0.04, 0)

    out["heuristic_score"] = score.clip(0.01, 0.80)

    ret_col = get_return_col(out)
    out["future_5d_return"] = out[ret_col]
    return out


def simulate(scores, score_col, name, threshold=0.30):
    df = scores.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "ticker", "close", "future_5d_return", score_col])
    df = df[df["date"] >= pd.Timestamp(START_DATE)]
    df = df.sort_values(["date", score_col], ascending=[True, False])

    dates = sorted(df["date"].unique())
    rebal_dates = dates[::HOLD_DAYS]

    equity = INITIAL_CAPITAL
    equity_records = []
    trades = []

    for dt in rebal_dates:
        day = df[df["date"] == dt].copy()
        day = day[day[score_col] >= threshold]
        day = day.sort_values(score_col, ascending=False).head(TOP_N)

        if day.empty:
            equity_records.append({"date": dt, "equity": equity, "asset": name})
            continue

        running_cost = 0.0
        period_pnl = 0.0

        for _, row in day.iterrows():
            remaining = equity - running_cost
            if remaining <= 0:
                break

            target_cost = min(equity * MAX_POSITION_PCT, remaining)

            if target_cost < equity * MIN_POSITION_PCT:
                continue

            entry_price = float(row["close"]) * (1 + SLIPPAGE_BPS / 10000)
            shares = int((target_cost - MIN_COMMISSION) / entry_price)

            while shares > 0:
                buy_commission = commission(shares)
                entry_value = shares * entry_price
                entry_total_cost = entry_value + buy_commission

                if entry_total_cost <= remaining and entry_total_cost >= equity * MIN_POSITION_PCT:
                    break
                shares -= 1

            if shares <= 0:
                continue

            fwd_ret = float(row["future_5d_return"])
            exit_price = entry_price * (1 + fwd_ret) * (1 - SLIPPAGE_BPS / 10000)
            sell_commission = commission(shares)
            exit_value = shares * exit_price - sell_commission
            pnl = exit_value - entry_total_cost

            running_cost += entry_total_cost
            period_pnl += pnl

            trades.append({
                "asset": name,
                "date": dt,
                "ticker": row["ticker"],
                "score": float(row[score_col]),
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_cost": entry_total_cost,
                "exit_value": exit_value,
                "commission": buy_commission + sell_commission,
                "future_5d_return_pct": fwd_ret * 100,
                "net_pnl": pnl,
                "net_return_pct": pnl / entry_total_cost * 100,
            })

        equity += period_pnl
        equity_records.append({"date": dt, "equity": equity, "asset": name})

    return pd.DataFrame(equity_records), pd.DataFrame(trades)


def spy_equity(start, end):
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if spy.empty:
        return pd.DataFrame(columns=["date", "equity", "asset"])

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]

    spy = spy.reset_index()
    spy.columns = [str(c).lower() for c in spy.columns]
    date_col = "date" if "date" in spy.columns else "datetime"
    spy = spy[[date_col, "close"]].rename(columns={date_col: "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy["equity"] = INITIAL_CAPITAL * spy["close"] / spy["close"].iloc[0]
    spy["asset"] = "SPY"
    return spy[["date", "equity", "asset"]]


def metrics_for_equity(eq, asset):
    g = eq.sort_values("date").copy()
    g["return"] = g["equity"].pct_change()
    mdd, _ = max_drawdown(g["equity"])
    cg = cagr(g["equity"], g["date"])
    return {
        "asset": asset,
        "start": g["date"].iloc[0].date(),
        "end": g["date"].iloc[-1].date(),
        "total_return_pct": (g["equity"].iloc[-1] / g["equity"].iloc[0] - 1) * 100,
        "cagr_pct": cg * 100,
        "volatility_pct": g["return"].std() * np.sqrt(252) * 100,
        "max_drawdown_pct": mdd * 100,
        "sharpe": sharpe_ratio(g["return"]),
        "sortino": sortino_ratio(g["return"]),
        "calmar": cg / abs(mdd) if mdd < 0 else np.nan,
    }


def overall_summary(equity, trades):
    rows = []
    for asset, g in equity.groupby("asset"):
        row = metrics_for_equity(g, asset)
        if asset != "SPY" and not trades.empty:
            tg = trades[trades["asset"] == asset]
            wins = tg[tg["net_pnl"] > 0]
            losses = tg[tg["net_pnl"] <= 0]
            row.update({
                "trades": len(tg),
                "win_rate_pct": (tg["net_pnl"] > 0).mean() * 100 if len(tg) else np.nan,
                "avg_trade_pct": tg["net_return_pct"].mean() if len(tg) else np.nan,
                "profit_factor": wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()) if len(losses) and losses["net_pnl"].sum() != 0 else np.nan,
                "total_commissions": tg["commission"].sum() if len(tg) else np.nan,
            })
        else:
            row.update({"trades": 0, "win_rate_pct": np.nan, "avg_trade_pct": np.nan, "profit_factor": np.nan, "total_commissions": np.nan})
        rows.append(row)

    out = pd.DataFrame(rows)

    # Add ML minus SPY and ML minus heuristic rows where possible.
    def diff_row(a, b, label):
        aa = out[out["asset"] == a]
        bb = out[out["asset"] == b]
        if aa.empty or bb.empty:
            return None
        aa = aa.iloc[0]
        bb = bb.iloc[0]
        return {
            "asset": label,
            "start": np.nan,
            "end": np.nan,
            "total_return_pct": aa["total_return_pct"] - bb["total_return_pct"],
            "cagr_pct": aa["cagr_pct"] - bb["cagr_pct"],
            "volatility_pct": aa["volatility_pct"] - bb["volatility_pct"],
            "max_drawdown_pct": aa["max_drawdown_pct"] - bb["max_drawdown_pct"],
            "sharpe": aa["sharpe"] - bb["sharpe"],
            "sortino": aa["sortino"] - bb["sortino"],
            "calmar": aa["calmar"] - bb["calmar"],
            "trades": aa.get("trades", np.nan),
            "win_rate_pct": aa.get("win_rate_pct", np.nan),
            "avg_trade_pct": aa.get("avg_trade_pct", np.nan),
            "profit_factor": aa.get("profit_factor", np.nan),
            "total_commissions": aa.get("total_commissions", np.nan),
        }

    extras = []
    for item in [
        diff_row("ML Strategy", "SPY", "ML minus SPY"),
        diff_row("ML Strategy", "Heuristic Strategy", "ML minus Heuristic"),
    ]:
        if item is not None:
            extras.append(item)

    if extras:
        out = pd.concat([out, pd.DataFrame(extras)], ignore_index=True)

    return out


def yearly_summary(equity):
    rows = []
    for asset, g in equity.groupby("asset"):
        g = g.sort_values("date").copy()
        g["year"] = g["date"].dt.year
        for year, yg in g.groupby("year"):
            mdd, _ = max_drawdown(yg["equity"])
            rows.append({
                "year": year,
                "asset": asset,
                "return_pct": (yg["equity"].iloc[-1] / yg["equity"].iloc[0] - 1) * 100,
                "max_drawdown_pct": mdd * 100,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    pivot = out.pivot(index="year", columns="asset", values="return_pct").reset_index()
    if "ML Strategy" in pivot.columns and "SPY" in pivot.columns:
        pivot["ml_minus_spy_pct"] = pivot["ML Strategy"] - pivot["SPY"]
        pivot["ml_beat_spy"] = pivot["ml_minus_spy_pct"] > 0
    if "ML Strategy" in pivot.columns and "Heuristic Strategy" in pivot.columns:
        pivot["ml_minus_heuristic_pct"] = pivot["ML Strategy"] - pivot["Heuristic Strategy"]
        pivot["ml_beat_heuristic"] = pivot["ml_minus_heuristic_pct"] > 0
    return pivot


def regime_summary(equity, trades):
    rows = []
    end_default = equity["date"].max()

    for label, start, end in PERIODS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else end_default

        for asset, g in equity.groupby("asset"):
            sub = g[(g["date"] >= s) & (g["date"] <= e)].sort_values("date")
            if len(sub) < 2:
                continue
            row = metrics_for_equity(sub, asset)
            row["period"] = label
            row["period_start"] = sub["date"].iloc[0].date()
            row["period_end"] = sub["date"].iloc[-1].date()

            if asset != "SPY":
                tg = trades[(trades["asset"] == asset) & (trades["date"] >= s) & (trades["date"] <= e)]
                row["trades"] = len(tg)
                row["win_rate_pct"] = (tg["net_pnl"] > 0).mean() * 100 if len(tg) else np.nan
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Compact comparison by regime.
    compact_rows = []
    for period, g in out.groupby("period"):
        row = {"period": period}
        for asset in ["ML Strategy", "Heuristic Strategy", "SPY"]:
            ag = g[g["asset"] == asset]
            if ag.empty:
                continue
            ag = ag.iloc[0]
            prefix = asset.lower().replace(" ", "_")
            row[f"{prefix}_return_pct"] = ag["total_return_pct"]
            row[f"{prefix}_cagr_pct"] = ag["cagr_pct"]
            row[f"{prefix}_max_drawdown_pct"] = ag["max_drawdown_pct"]
            row[f"{prefix}_sharpe"] = ag["sharpe"]
        if "ml_strategy_return_pct" in row and "spy_return_pct" in row:
            row["ml_minus_spy_pct"] = row["ml_strategy_return_pct"] - row["spy_return_pct"]
            row["ml_beat_spy"] = row["ml_minus_spy_pct"] > 0
        if "ml_strategy_return_pct" in row and "heuristic_strategy_return_pct" in row:
            row["ml_minus_heuristic_pct"] = row["ml_strategy_return_pct"] - row["heuristic_strategy_return_pct"]
            row["ml_beat_heuristic"] = row["ml_minus_heuristic_pct"] > 0
        compact_rows.append(row)

    return pd.DataFrame(compact_rows)


def make_charts(equity, yearly):
    files = []

    plt.figure(figsize=(12, 6))
    for asset, g in equity.groupby("asset"):
        g = g.sort_values("date")
        plt.plot(g["date"], g["equity"], label=asset)
    plt.title("Equity Curve: ML Strategy vs Heuristic vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    f = "ml_period_analysis_equity_curve.png"
    plt.savefig(f, dpi=160)
    plt.close()
    files.append(f)

    plt.figure(figsize=(12, 6))
    for asset, g in equity.groupby("asset"):
        g = g.sort_values("date")
        _, dd = max_drawdown(g["equity"])
        plt.plot(g["date"], dd * 100, label=asset)
    plt.title("Drawdown: ML Strategy vs Heuristic vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    f = "ml_period_analysis_drawdown.png"
    plt.savefig(f, dpi=160)
    plt.close()
    files.append(f)

    if not yearly.empty:
        cols = [c for c in ["ML Strategy", "Heuristic Strategy", "SPY"] if c in yearly.columns]
        if cols:
            ax = yearly.set_index("year")[cols].plot(kind="bar", figsize=(12, 6))
            ax.set_title("Yearly Returns")
            ax.set_ylabel("Return (%)")
            plt.tight_layout()
            f = "ml_period_analysis_yearly_returns.png"
            plt.savefig(f, dpi=160)
            plt.close()
            files.append(f)

    return files


def html_report(overall, yearly, regimes, chart_files):
    html = []
    html.append("<html><head><title>ML Period Analysis</title>")
    html.append("""
<style>
body { font-family: Arial, sans-serif; margin: 40px; color: #222; }
table { border-collapse: collapse; margin: 20px 0; width: 100%; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 7px; text-align: right; }
th { background: #f3f3f3; text-align: center; }
td:first-child, th:first-child { text-align: left; }
h1, h2 { color: #111; }
img { max-width: 100%; margin: 16px 0 32px 0; border: 1px solid #ddd; }
.note { background: #fff8dc; padding: 12px; border-left: 4px solid #e0b000; margin: 20px 0; }
</style>
""")
    html.append("</head><body>")
    html.append("<h1>ML Period Analysis: ML Strategy vs Heuristic Strategy vs SPY</h1>")
    html.append(f"<p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append("<div class='note'><b>Important:</b> This is a research backtest, not proof of future performance. Review data quality, execution assumptions, slippage, and survivorship bias before using real capital.</div>")

    html.append("<h2>Overall Performance</h2>")
    html.append(overall.to_html(index=False, float_format=lambda x: f"{x:,.2f}"))

    html.append("<h2>Year-by-Year Performance</h2>")
    html.append(yearly.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not yearly.empty else "<p>No yearly results.</p>")

    html.append("<h2>Market Regime Performance</h2>")
    html.append(regimes.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not regimes.empty else "<p>No regime results.</p>")

    html.append("<h2>Charts</h2>")
    for f in chart_files:
        html.append(f"<h3>{f}</h3>")
        html.append(f"<img src='{f}' />")

    html.append("</body></html>")

    with open("ml_period_analysis_report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def main():
    print("=" * 78)
    print("ML PERIOD ANALYSIS: ML STRATEGY VS HEURISTIC VS SPY")
    print("=" * 78)

    ml, heuristic = load_data()

    print("Simulating ML strategy...")
    ml_equity, ml_trades = simulate(ml, "ml_score", "ML Strategy", threshold=0.30)

    print("Simulating heuristic strategy...")
    h_equity, h_trades = simulate(heuristic, "heuristic_score", "Heuristic Strategy", threshold=0.30)

    start = min(ml_equity["date"].min(), h_equity["date"].min())
    end = max(ml_equity["date"].max(), h_equity["date"].max())

    print("Downloading SPY benchmark...")
    spy = spy_equity(start, end)

    equity = pd.concat([ml_equity, h_equity, spy], ignore_index=True)
    trades = pd.concat([ml_trades, h_trades], ignore_index=True)

    overall = overall_summary(equity, trades)
    yearly = yearly_summary(equity)
    regimes = regime_summary(equity, trades)

    chart_files = make_charts(equity, yearly)
    html_report(overall, yearly, regimes, chart_files)

    equity.to_csv("ml_period_analysis_equity.csv", index=False)
    trades.to_csv("ml_period_analysis_trades.csv", index=False)
    overall.to_csv("ml_period_analysis_overall.csv", index=False)
    yearly.to_csv("ml_period_analysis_yearly.csv", index=False)
    regimes.to_csv("ml_period_analysis_regimes.csv", index=False)

    print("\n" + "=" * 78)
    print("OVERALL PERFORMANCE")
    print("=" * 78)
    print(overall.to_string(index=False))

    print("\n" + "=" * 78)
    print("YEAR-BY-YEAR PERFORMANCE")
    print("=" * 78)
    print(yearly.to_string(index=False))

    print("\n" + "=" * 78)
    print("MARKET REGIME PERFORMANCE")
    print("=" * 78)
    print(regimes.to_string(index=False))

    print("\nSaved:")
    print("  ml_period_analysis_overall.csv")
    print("  ml_period_analysis_yearly.csv")
    print("  ml_period_analysis_regimes.csv")
    print("  ml_period_analysis_equity.csv")
    print("  ml_period_analysis_trades.csv")
    print("  ml_period_analysis_report.html")
    for f in chart_files:
        print(f"  {f}")
    print("\nOpen ml_period_analysis_report.html for the full visual report.")


if __name__ == "__main__":
    main()
