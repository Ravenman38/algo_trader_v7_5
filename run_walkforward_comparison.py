#!/usr/bin/env python3
"""
AlgoTrader v5: rigorous walk-forward portfolio comparison.

This script compares:
  1. ML Strategy (uses out-of-sample walk-forward predictions from train_probability_model.py)
  2. Heuristic Strategy (same portfolio rules, same dates)
  3. SPY benchmark (same dates)

Important design choices:
  - ML predictions are out-of-sample by test year.
  - All headline results are aligned to the same date range.
  - Commissions and slippage are included.
  - Results are broken down by year and market regime.
  - A clear HTML report and PNG charts are produced.

Run:
  python run.py train-ml --start 2018-01-01 --model gbdt
  python run.py walkforward

Outputs:
  walkforward_comparison_report.html
  walkforward_comparison_overall.csv
  walkforward_comparison_yearly.csv
  walkforward_comparison_regimes.csv
  walkforward_comparison_equity.csv
  walkforward_comparison_trades.csv
  walkforward_equity_curve.png
  walkforward_drawdown.png
  walkforward_yearly_returns.png
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
HOLD_DAYS = 5
TOP_N = 12
MAX_POSITION_PCT = 0.15
MIN_POSITION_PCT = 0.02
PROB_THRESHOLD = 0.30
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00
SLIPPAGE_BPS = 5

ML_PREDICTIONS_FILE = "ml_probability_predictions.csv"
TRAINING_DATA_FILE = "probability_training_dataset.csv"

PERIODS = [
    ("2021 bull / reopening", "2021-01-01", "2021-12-31"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
    ("2023 recovery", "2023-01-01", "2023-12-31"),
    ("2024 bull / AI rally", "2024-01-01", "2024-12-31"),
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
    raise ValueError(f"Could not find future 5-day return column. Available columns: {list(df.columns)[:40]}...")


def normalize_prob(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.max() > 1.5:
        s = s / 100.0
    return s


def load_ml_and_training_data():
    if not os.path.exists(ML_PREDICTIONS_FILE):
        raise FileNotFoundError(
            f"{ML_PREDICTIONS_FILE} not found. Run: python run.py train-ml --start 2018-01-01 --model gbdt"
        )
    if not os.path.exists(TRAINING_DATA_FILE):
        raise FileNotFoundError(
            f"{TRAINING_DATA_FILE} not found. Run: python run.py train-ml --start 2018-01-01 --model gbdt"
        )

    ml = pd.read_csv(ML_PREDICTIONS_FILE)
    data = pd.read_csv(TRAINING_DATA_FILE)
    ml["date"] = pd.to_datetime(ml["date"])
    data["date"] = pd.to_datetime(data["date"])

    prob_col = None
    for c in ["ml_prob_up_5pct", "ml_prob", "prob_up_5pct", "pred_prob", "probability"]:
        if c in ml.columns:
            prob_col = c
            break
    if prob_col is None:
        raise ValueError("Could not find ML probability column in ml_probability_predictions.csv")

    ml["ml_score"] = normalize_prob(ml[prob_col])

    ret_col = get_return_col(data)
    if "future_5d_return" not in ml.columns:
        ml = ml.merge(
            data[["date", "ticker", ret_col]].rename(columns={ret_col: "future_5d_return"}),
            on=["date", "ticker"],
            how="left",
        )
    if "close" not in ml.columns:
        ml = ml.merge(data[["date", "ticker", "close"]], on=["date", "ticker"], how="left")

    heuristic = build_heuristic_scores(data.copy())
    return ml, heuristic


def build_heuristic_scores(df):
    if "target_up_5pct" in df.columns:
        base = float(pd.to_numeric(df["target_up_5pct"], errors="coerce").mean())
    else:
        base = 0.12

    score = pd.Series(base, index=df.index, dtype=float)

    def q(col, p):
        return pd.to_numeric(df[col], errors="coerce").quantile(p)

    if "ret_21d" in df.columns:
        score += np.where(df["ret_21d"] > q("ret_21d", 0.70), 0.05, 0)
    if "ret_63d" in df.columns:
        score += np.where(df["ret_63d"] > q("ret_63d", 0.70), 0.05, 0)
    if "dist_ma_20d" in df.columns:
        score += np.where(df["dist_ma_20d"] > 0, 0.03, 0)
    if "dist_ma_50d" in df.columns:
        score += np.where(df["dist_ma_50d"] > 0, 0.03, 0)
    if "dist_52w_high" in df.columns:
        score += np.where(df["dist_52w_high"] > -0.15, 0.04, 0)
    if "rsi_14d" in df.columns:
        score += np.where((df["rsi_14d"] >= 45) & (df["rsi_14d"] <= 75), 0.03, 0)
    if "spy_above_200d" in df.columns:
        score += np.where(df["spy_above_200d"] > 0, 0.04, -0.04)
    if "vol_21d" in df.columns:
        score += np.where((df["vol_21d"] >= 0.20) & (df["vol_21d"] <= 0.80), 0.04, 0)

    df["heuristic_score"] = score.clip(0.01, 0.80)
    df["future_5d_return"] = pd.to_numeric(df[get_return_col(df)], errors="coerce")
    return df


def simulate(scores, score_col, asset_name, start_date=None, end_date=None):
    df = scores.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["future_5d_return"] = pd.to_numeric(df["future_5d_return"], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close", "future_5d_return", score_col])

    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    df = df.sort_values(["date", score_col], ascending=[True, False])
    dates = sorted(df["date"].unique())
    rebal_dates = dates[::HOLD_DAYS]

    equity = INITIAL_CAPITAL
    equity_records = []
    trades = []

    for dt in rebal_dates:
        day = df[df["date"] == dt].copy()
        day = day[day[score_col] >= PROB_THRESHOLD]
        day = day.sort_values(score_col, ascending=False).head(TOP_N)

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
                buy_comm = commission(shares)
                entry_value = shares * entry_price
                entry_total_cost = entry_value + buy_comm
                if entry_total_cost <= remaining and entry_total_cost >= equity * MIN_POSITION_PCT:
                    break
                shares -= 1

            if shares <= 0:
                continue

            fwd_ret = float(row["future_5d_return"])
            exit_price = entry_price * (1 + fwd_ret) * (1 - SLIPPAGE_BPS / 10000)
            sell_comm = commission(shares)
            exit_value = shares * exit_price - sell_comm
            pnl = exit_value - entry_total_cost

            running_cost += entry_total_cost
            period_pnl += pnl

            trades.append({
                "asset": asset_name,
                "date": dt,
                "ticker": row["ticker"],
                "score": float(row[score_col]),
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_cost": entry_total_cost,
                "exit_value": exit_value,
                "commission": buy_comm + sell_comm,
                "future_5d_return_pct": fwd_ret * 100,
                "net_pnl": pnl,
                "net_return_pct": pnl / entry_total_cost * 100 if entry_total_cost else np.nan,
            })

        equity += period_pnl
        equity_records.append({"date": dt, "equity": equity, "asset": asset_name})

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


def align_to_common_period(equity, trades):
    starts = equity.groupby("asset")["date"].min()
    ends = equity.groupby("asset")["date"].max()
    common_start = starts.max()
    common_end = ends.min()

    aligned = []
    for asset, g in equity.groupby("asset"):
        g = g[(g["date"] >= common_start) & (g["date"] <= common_end)].sort_values("date").copy()
        if g.empty:
            continue
        # Rebase all series to $100,000 at common_start for fair comparison.
        g["equity"] = INITIAL_CAPITAL * g["equity"] / g["equity"].iloc[0]
        aligned.append(g)

    eq_aligned = pd.concat(aligned, ignore_index=True)
    tr_aligned = trades[(trades["date"] >= common_start) & (trades["date"] <= common_end)].copy()
    return eq_aligned, tr_aligned, common_start, common_end


def metrics_for_equity(g, asset):
    g = g.sort_values("date").copy()
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
        if asset != "SPY":
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

    extras = [
        diff_row("ML Strategy", "SPY", "ML minus SPY"),
        diff_row("ML Strategy", "Heuristic Strategy", "ML minus Heuristic"),
    ]
    extras = [x for x in extras if x is not None]
    if extras:
        out = pd.concat([out, pd.DataFrame(extras)], ignore_index=True)
    return out


def yearly_summary(equity):
    rows = []
    for asset, g in equity.groupby("asset"):
        g = g.sort_values("date").copy()
        g["year"] = g["date"].dt.year
        for year, yg in g.groupby("year"):
            if len(yg) < 2:
                continue
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
    common_end = equity["date"].max()
    for label, start, end in PERIODS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else common_end
        row = {"period": label}
        for asset, g in equity.groupby("asset"):
            sub = g[(g["date"] >= s) & (g["date"] <= e)].sort_values("date")
            if len(sub) < 2:
                continue
            m = metrics_for_equity(sub, asset)
            prefix = asset.lower().replace(" ", "_")
            row[f"{prefix}_return_pct"] = m["total_return_pct"]
            row[f"{prefix}_cagr_pct"] = m["cagr_pct"]
            row[f"{prefix}_max_drawdown_pct"] = m["max_drawdown_pct"]
            row[f"{prefix}_sharpe"] = m["sharpe"]
            if asset != "SPY":
                tg = trades[(trades["asset"] == asset) & (trades["date"] >= s) & (trades["date"] <= e)]
                row[f"{prefix}_trades"] = len(tg)
        if "ml_strategy_return_pct" in row and "spy_return_pct" in row:
            row["ml_minus_spy_pct"] = row["ml_strategy_return_pct"] - row["spy_return_pct"]
            row["ml_beat_spy"] = row["ml_minus_spy_pct"] > 0
        if "ml_strategy_return_pct" in row and "heuristic_strategy_return_pct" in row:
            row["ml_minus_heuristic_pct"] = row["ml_strategy_return_pct"] - row["heuristic_strategy_return_pct"]
            row["ml_beat_heuristic"] = row["ml_minus_heuristic_pct"] > 0
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(overall, yearly):
    ml = overall[overall["asset"] == "ML Strategy"]
    spy = overall[overall["asset"] == "SPY"]
    heu = overall[overall["asset"] == "Heuristic Strategy"]
    if ml.empty or spy.empty:
        return "Insufficient data to compare ML against SPY."
    ml = ml.iloc[0]
    spy = spy.iloc[0]
    lines = []
    lines.append("DID ML BEAT SPY? " + ("YES" if ml["cagr_pct"] > spy["cagr_pct"] and ml["sharpe"] > spy["sharpe"] else "MIXED/NO"))
    lines.append(f"ML CAGR minus SPY: {ml['cagr_pct'] - spy['cagr_pct']:.2f} percentage points")
    lines.append(f"ML total return minus SPY: {ml['total_return_pct'] - spy['total_return_pct']:.2f} percentage points")
    lines.append(f"ML Sharpe minus SPY: {ml['sharpe'] - spy['sharpe']:.2f}")
    lines.append(f"ML max drawdown: {ml['max_drawdown_pct']:.2f}% vs SPY {spy['max_drawdown_pct']:.2f}%")
    if not heu.empty:
        h = heu.iloc[0]
        lines.append(f"ML CAGR minus Heuristic: {ml['cagr_pct'] - h['cagr_pct']:.2f} percentage points")
    if not yearly.empty and "ml_beat_spy" in yearly.columns:
        n = yearly["ml_beat_spy"].sum()
        d = yearly["ml_beat_spy"].notna().sum()
        lines.append(f"ML beat SPY in {int(n)} of {int(d)} calendar years in the aligned test period")
    return "\n".join(lines)


def make_charts(equity, yearly):
    files = []
    plt.figure(figsize=(12, 6))
    for asset, g in equity.groupby("asset"):
        g = g.sort_values("date")
        plt.plot(g["date"], g["equity"], label=asset)
    plt.title("Aligned Walk-Forward Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    f = "walkforward_equity_curve.png"
    plt.savefig(f, dpi=160)
    plt.close()
    files.append(f)

    plt.figure(figsize=(12, 6))
    for asset, g in equity.groupby("asset"):
        g = g.sort_values("date")
        _, dd = max_drawdown(g["equity"])
        plt.plot(g["date"], dd * 100, label=asset)
    plt.title("Aligned Walk-Forward Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    f = "walkforward_drawdown.png"
    plt.savefig(f, dpi=160)
    plt.close()
    files.append(f)

    cols = [c for c in ["ML Strategy", "Heuristic Strategy", "SPY"] if c in yearly.columns]
    if cols:
        ax = yearly.set_index("year")[cols].plot(kind="bar", figsize=(12, 6))
        ax.set_title("Aligned Walk-Forward Yearly Returns")
        ax.set_ylabel("Return (%)")
        plt.tight_layout()
        f = "walkforward_yearly_returns.png"
        plt.savefig(f, dpi=160)
        plt.close()
        files.append(f)
    return files


def make_html(overall, yearly, regimes, chart_files, verdict_text, common_start, common_end):
    css = """
<style>
body { font-family: Arial, sans-serif; margin: 40px; color: #222; }
table { border-collapse: collapse; margin: 20px 0; width: 100%; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 7px; text-align: right; }
th { background: #f3f3f3; text-align: center; }
td:first-child, th:first-child { text-align: left; }
h1, h2 { color: #111; }
img { max-width: 100%; margin: 16px 0 32px 0; border: 1px solid #ddd; }
.verdict { white-space: pre-wrap; background: #eef7ee; padding: 16px; border-left: 5px solid #2e7d32; font-family: monospace; }
.note { background: #fff8dc; padding: 12px; border-left: 4px solid #e0b000; margin: 20px 0; }
</style>
"""
    html = ["<html><head><title>AlgoTrader v5 Walk-Forward Comparison</title>", css, "</head><body>"]
    html.append("<h1>AlgoTrader v5 Walk-Forward Comparison</h1>")
    html.append(f"<p><b>Aligned test period:</b> {pd.Timestamp(common_start).date()} to {pd.Timestamp(common_end).date()}</p>")
    html.append(f"<p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append("<div class='note'>This report aligns ML Strategy, Heuristic Strategy, and SPY to the same test dates. ML predictions are out-of-sample by test year. This is still research, not proof of future returns.</div>")
    html.append("<h2>Bottom Line</h2>")
    html.append(f"<div class='verdict'>{verdict_text}</div>")
    html.append("<h2>Overall Performance</h2>")
    html.append(overall.to_html(index=False, float_format=lambda x: f"{x:,.2f}"))
    html.append("<h2>Year-by-Year Performance</h2>")
    html.append(yearly.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not yearly.empty else "<p>No yearly data.</p>")
    html.append("<h2>Market Regime Performance</h2>")
    html.append(regimes.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not regimes.empty else "<p>No regime data.</p>")
    html.append("<h2>Charts</h2>")
    for f in chart_files:
        html.append(f"<h3>{f}</h3><img src='{f}' />")
    html.append("</body></html>")
    with open("walkforward_comparison_report.html", "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))


def main():
    print("=" * 80)
    print("ALGOTRADER V5: ALIGNED WALK-FORWARD COMPARISON")
    print("ML Strategy vs Heuristic Strategy vs SPY")
    print("=" * 80)

    ml, heuristic = load_ml_and_training_data()

    ml_start = ml["date"].min()
    ml_end = ml["date"].max()
    print(f"ML out-of-sample prediction window: {ml_start.date()} to {ml_end.date()}")

    print("Simulating ML Strategy...")
    ml_eq, ml_tr = simulate(ml, "ml_score", "ML Strategy", start_date=ml_start, end_date=ml_end)

    print("Simulating Heuristic Strategy on same prediction window...")
    h_eq, h_tr = simulate(heuristic, "heuristic_score", "Heuristic Strategy", start_date=ml_start, end_date=ml_end)

    print("Downloading SPY benchmark...")
    spy = spy_equity(ml_start, ml_end)

    equity_raw = pd.concat([ml_eq, h_eq, spy], ignore_index=True)
    trades_raw = pd.concat([ml_tr, h_tr], ignore_index=True)

    equity, trades, common_start, common_end = align_to_common_period(equity_raw, trades_raw)
    print(f"Aligned comparison window: {pd.Timestamp(common_start).date()} to {pd.Timestamp(common_end).date()}")

    overall = overall_summary(equity, trades)
    yearly = yearly_summary(equity)
    regimes = regime_summary(equity, trades)
    verdict_text = verdict(overall, yearly)
    chart_files = make_charts(equity, yearly)
    make_html(overall, yearly, regimes, chart_files, verdict_text, common_start, common_end)

    overall.to_csv("walkforward_comparison_overall.csv", index=False)
    yearly.to_csv("walkforward_comparison_yearly.csv", index=False)
    regimes.to_csv("walkforward_comparison_regimes.csv", index=False)
    equity.to_csv("walkforward_comparison_equity.csv", index=False)
    trades.to_csv("walkforward_comparison_trades.csv", index=False)

    print("\n" + "=" * 80)
    print("BOTTOM LINE")
    print("=" * 80)
    print(verdict_text)

    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    print(overall.to_string(index=False))

    print("\n" + "=" * 80)
    print("YEAR-BY-YEAR PERFORMANCE")
    print("=" * 80)
    print(yearly.to_string(index=False))

    print("\nSaved:")
    print("  walkforward_comparison_report.html")
    print("  walkforward_comparison_overall.csv")
    print("  walkforward_comparison_yearly.csv")
    print("  walkforward_comparison_regimes.csv")
    print("  walkforward_comparison_equity.csv")
    print("  walkforward_comparison_trades.csv")
    for f in chart_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
