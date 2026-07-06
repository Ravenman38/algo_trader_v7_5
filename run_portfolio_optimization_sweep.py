#!/usr/bin/env python3
"""
AlgoTrader v7.4: Fast Portfolio Optimizer Sweep with Idle-Cash Yield

This script DOES NOT retrain the ML model. It reuses:
  - ml_probability_predictions.csv
  - probability_training_dataset.csv

It tests many portfolio-construction settings quickly, ranks them, and writes:
  - portfolio_optimization_sweep_report.html
  - portfolio_optimization_sweep_results.csv
  - portfolio_construction_report.html  (best configuration detail report)

Run after training once:
  python run.py optimize-portfolio
"""

import argparse
import itertools
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import run_portfolio_construction as pc


DD_MODES = {
    # No drawdown brake. Useful to show whether vol/correlation controls alone work.
    "none": [(0.00, 1.00), (-1.00, 1.00)],
    # Mild brake: still lets the strategy recover, but stops full risk in deeper drawdowns.
    "mild": [(0.00, 1.00), (-0.15, 0.90), (-0.25, 0.75), (-0.35, 0.60), (-1.00, 0.45)],
    # Medium brake: close to v7 but less restrictive at deeper levels.
    "medium": [(0.00, 1.00), (-0.15, 0.80), (-0.25, 0.60), (-0.35, 0.40), (-1.00, 0.30)],
}


def install_drawdown_mode(mode: str):
    rules = DD_MODES[mode]

    def _cap(current_equity, peak_equity):
        dd = current_equity / peak_equity - 1 if peak_equity > 0 else 0.0
        for threshold, cap in rules:
            if dd >= threshold:
                return cap
        return rules[-1][1]

    pc.drawdown_exposure_cap = _cap


def set_config(cfg: dict):
    pc.PROB_THRESHOLD = cfg["prob_threshold"]
    pc.TARGET_PORTFOLIO_VOL = cfg["target_vol"]
    pc.MAX_GROSS_EXPOSURE = cfg["max_exposure"]
    pc.MIN_GROSS_EXPOSURE = cfg["min_exposure"]
    pc.OPTIMIZER_MAX_POSITION_PCT = cfg["max_position"]
    pc.MAX_PAIRWISE_CORR = cfg["max_corr"]
    pc.SLIPPAGE_BPS = cfg["slippage_bps"]
    pc.OPTIMIZER_MAX_HOLDINGS = cfg["max_holdings"]
    install_drawdown_mode(cfg["drawdown_mode"])


def summarize_opt(equity, trades, allocs, baseline_row=None, spy_row=None):
    overall = pc.overall_summary(equity, trades)
    opt = overall[overall["asset"] == "ML Optimized"]
    if opt.empty:
        return None
    row = opt.iloc[0].to_dict()
    if baseline_row is not None:
        row["cagr_vs_baseline"] = row["cagr_pct"] - baseline_row["cagr_pct"]
        row["dd_improvement_vs_baseline"] = row["max_drawdown_pct"] - baseline_row["max_drawdown_pct"]
        row["sharpe_vs_baseline"] = row["sharpe"] - baseline_row["sharpe"]
    if spy_row is not None:
        row["cagr_vs_spy"] = row["cagr_pct"] - spy_row["cagr_pct"]
        row["dd_vs_spy"] = row["max_drawdown_pct"] - spy_row["max_drawdown_pct"]
        row["sharpe_vs_spy"] = row["sharpe"] - spy_row["sharpe"]
    if not allocs.empty:
        oa = allocs[allocs["asset"] == "ML Optimized"].copy()
        row["avg_gross_exposure_pct"] = oa["gross_exposure"].mean() * 100 if len(oa) else np.nan
        row["avg_holdings"] = oa["holdings"].mean() if len(oa) else np.nan
        row["avg_estimated_vol_pct"] = oa["estimated_vol"].mean() * 100 if len(oa) else np.nan
    return row


def score_config(row):
    """Rank by practical deployability, not maximum return."""
    cagr = row.get("cagr_pct", np.nan)
    dd = row.get("max_drawdown_pct", np.nan)
    sharpe = row.get("sharpe", np.nan)
    calmar = row.get("calmar", np.nan)
    spy_edge = row.get("cagr_vs_spy", np.nan)
    exposure = row.get("avg_gross_exposure_pct", np.nan)

    if pd.isna(cagr) or pd.isna(dd) or pd.isna(sharpe):
        return -999

    # Hard penalties for failing the main objective.
    penalty = 0.0
    if cagr < 14.32:
        penalty += 10.0
    if dd < -45:
        penalty += abs(dd + 45) * 0.35
    if exposure < 50:
        penalty += (50 - exposure) * 0.08
    if exposure > 85:
        penalty += (exposure - 85) * 0.08

    return (
        0.45 * cagr
        + 12.0 * sharpe
        + 6.0 * max(calmar, -1)
        + 0.35 * dd          # less negative drawdown is better
        + 0.60 * spy_edge
        - penalty
    )


def build_configs(mode: str):
    if mode == "quick":
        grid = {
            "target_vol": [0.28, 0.32, 0.36],
            "max_exposure": [0.85, 1.00],
            "min_exposure": [0.20, 0.30],
            "max_position": [0.10, 0.12, 0.15],
            "max_corr": [0.75, 0.85],
            "drawdown_mode": ["mild", "medium"],
            "prob_threshold": [0.30],
            "slippage_bps": [10],
            "max_holdings": [16],
        }
    else:
        grid = {
            "target_vol": [0.24, 0.28, 0.32, 0.36, 0.40],
            "max_exposure": [0.75, 0.85, 1.00],
            "min_exposure": [0.10, 0.20, 0.30],
            "max_position": [0.08, 0.10, 0.12, 0.15],
            "max_corr": [0.70, 0.80, 0.90],
            "drawdown_mode": ["none", "mild", "medium"],
            "prob_threshold": [0.28, 0.30, 0.32],
            "slippage_bps": [10],
            "max_holdings": [16, 20],
        }
    keys = list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def make_sweep_charts(results):
    files = []
    if results.empty:
        return files

    plt.figure(figsize=(10, 6))
    plt.scatter(results["max_drawdown_pct"], results["cagr_pct"], s=22)
    plt.axhline(14.32, linestyle="--", linewidth=1)
    plt.axvline(-40, linestyle="--", linewidth=1)
    plt.title("Optimizer Sweep: CAGR vs Max Drawdown")
    plt.xlabel("Max Drawdown (%)")
    plt.ylabel("CAGR (%)")
    plt.tight_layout()
    f = "portfolio_optimization_sweep_cagr_vs_drawdown.png"
    plt.savefig(f, dpi=160)
    plt.close(); files.append(f)

    plt.figure(figsize=(10, 6))
    plt.scatter(results["avg_gross_exposure_pct"], results["cagr_pct"], s=22)
    plt.title("Optimizer Sweep: Exposure vs CAGR")
    plt.xlabel("Average Gross Exposure (%)")
    plt.ylabel("CAGR (%)")
    plt.tight_layout()
    f = "portfolio_optimization_sweep_exposure_vs_cagr.png"
    plt.savefig(f, dpi=160)
    plt.close(); files.append(f)

    return files


def make_sweep_html(results, baseline_row, spy_row, chart_files, mode):
    top = results.sort_values("optimizer_score", ascending=False).head(25).copy()
    robust = results[
        (results["cagr_pct"] > spy_row["cagr_pct"]) &
        (results["max_drawdown_pct"] > -45) &
        (results["sharpe"] >= 1.3)
    ].copy()

    best = top.iloc[0] if len(top) else None
    html = []
    html.append("<html><head><title>AlgoTrader v7.4 Portfolio Optimization Sweep</title>")
    html.append("""
<style>
body { font-family: Arial, sans-serif; margin: 40px; color: #222; }
table { border-collapse: collapse; margin: 20px 0; width: 100%; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 7px; text-align: right; }
th { background: #f3f3f3; text-align: center; }
td:first-child, th:first-child { text-align: left; }
.verdict { white-space: pre-wrap; background: #eef7ee; padding: 16px; border-left: 5px solid #2e7d32; font-family: monospace; }
.note { background: #fff8dc; padding: 12px; border-left: 4px solid #e0b000; margin: 20px 0; }
img { max-width: 100%; margin: 16px 0 32px 0; border: 1px solid #ddd; }
</style>
""")
    html.append("</head><body>")
    html.append("<h1>AlgoTrader v7.4 Portfolio Optimization Sweep</h1>")
    html.append(f"<p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append(f"<p><b>Sweep mode:</b> {mode}</p>")
    html.append("<div class='note'>This report reuses existing ML predictions, the original/simple SPY regime filter, and an idle-cash proxy. It does not retrain the model.</div>")

    verdict = []
    verdict.append(f"Configurations tested: {len(results)}")
    verdict.append(f"Robust configs: {len(robust)} where CAGR > SPY, drawdown > -45%, Sharpe >= 1.3")
    verdict.append(f"Baseline CAGR/DD/Sharpe: {baseline_row['cagr_pct']:.2f}% / {baseline_row['max_drawdown_pct']:.2f}% / {baseline_row['sharpe']:.2f}")
    verdict.append(f"SPY CAGR/DD/Sharpe: {spy_row['cagr_pct']:.2f}% / {spy_row['max_drawdown_pct']:.2f}% / {spy_row['sharpe']:.2f}")
    if best is not None:
        verdict.append("")
        verdict.append("Best practical config:")
        verdict.append(f"  CAGR: {best['cagr_pct']:.2f}%")
        verdict.append(f"  Max drawdown: {best['max_drawdown_pct']:.2f}%")
        verdict.append(f"  Sharpe: {best['sharpe']:.2f}")
        verdict.append(f"  Avg exposure: {best['avg_gross_exposure_pct']:.2f}%")
        verdict.append(f"  Settings: target_vol={best['target_vol']:.2f}, max_exposure={best['max_exposure']:.2f}, max_position={best['max_position']:.2f}, max_corr={best['max_corr']:.2f}, drawdown_mode={best['drawdown_mode']}")
    html.append("<h2>Bottom Line</h2>")
    html.append("<div class='verdict'>" + "\n".join(verdict) + "</div>")

    html.append("<h2>Top 25 Configurations</h2>")
    cols = [
        "rank", "optimizer_score", "cagr_pct", "max_drawdown_pct", "sharpe", "sortino", "calmar",
        "total_return_pct", "volatility_pct", "avg_gross_exposure_pct", "avg_holdings",
        "cagr_vs_spy", "dd_improvement_vs_baseline", "target_vol", "max_exposure", "min_exposure",
        "max_position", "max_corr", "drawdown_mode", "prob_threshold", "slippage_bps", "max_holdings"
    ]
    show_cols = [c for c in cols if c in top.columns]
    html.append(top[show_cols].to_html(index=False, float_format=lambda x: f"{x:,.2f}"))

    if len(robust):
        html.append("<h2>Robust Configurations</h2>")
        rshow = robust.sort_values("optimizer_score", ascending=False).head(50)
        html.append(rshow[show_cols].to_html(index=False, float_format=lambda x: f"{x:,.2f}"))
    else:
        html.append("<h2>Robust Configurations</h2><p>No configuration met the robust-config criteria.</p>")

    html.append("<h2>Charts</h2>")
    for f in chart_files:
        html.append(f"<h3>{f}</h3><img src='{f}' />")
    html.append("</body></html>")
    with open("portfolio_optimization_sweep_report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def save_best_detail(best_cfg, scores, data, base_eq, base_trades, base_alloc, spy, cash_returns=None):
    """Reuse v7 reporting functions to create portfolio_construction_report.html for best config."""
    set_config(best_cfg)
    opt_eq, opt_trades, opt_alloc = pc.simulate_optimized(scores, data, cash_returns)

    start = min(base_eq["date"].min(), opt_eq["date"].min())
    end = max(base_eq["date"].max(), opt_eq["date"].max())
    common_start = max(base_eq["date"].min(), opt_eq["date"].min(), spy["date"].min() if not spy.empty else start)
    common_end = min(base_eq["date"].max(), opt_eq["date"].max(), spy["date"].max() if not spy.empty else end)

    equity = pd.concat([base_eq, opt_eq, spy], ignore_index=True)
    equity = equity[(equity["date"] >= common_start) & (equity["date"] <= common_end)].copy()
    trades = pd.concat([base_trades, opt_trades], ignore_index=True)
    trades = trades[(trades["date"] >= common_start) & (trades["date"] <= common_end)].copy()
    allocs = pd.concat([base_alloc, opt_alloc], ignore_index=True)
    if not allocs.empty:
        allocs["date"] = pd.to_datetime(allocs["date"])
        allocs = allocs[(allocs["date"] >= common_start) & (allocs["date"] <= common_end)].copy()

    overall = pc.overall_summary(equity, trades)
    yearly = pc.yearly_summary(equity)
    regimes = pc.regime_summary(equity)
    if not allocs.empty:
        alloc_summary = allocs.groupby("asset").agg(
            avg_gross_exposure_pct=("gross_exposure", lambda x: x.mean() * 100),
            min_gross_exposure_pct=("gross_exposure", lambda x: x.min() * 100),
            max_gross_exposure_pct=("gross_exposure", lambda x: x.max() * 100),
            avg_deployed_pct=("deployed_pct", lambda x: x.mean() * 100),
            avg_idle_cash_pct=("idle_cash_pct", lambda x: x.mean() * 100),
            avg_cash_proxy_return_pct=("cash_period_return_pct", "mean"),
            total_cash_pnl=("cash_pnl", "sum"),
            avg_holdings=("holdings", "mean"),
            avg_estimated_vol_pct=("estimated_vol", lambda x: x.mean() * 100),
        ).reset_index()
    else:
        alloc_summary = pd.DataFrame()

    chart_files = pc.make_charts(equity, yearly, allocs)
    pc.make_html(overall, yearly, regimes, alloc_summary, chart_files)

    overall.to_csv("portfolio_construction_overall.csv", index=False)
    yearly.to_csv("portfolio_construction_yearly.csv", index=False)
    regimes.to_csv("portfolio_construction_regimes.csv", index=False)
    equity.to_csv("portfolio_construction_equity.csv", index=False)
    trades.to_csv("portfolio_construction_trades.csv", index=False)
    allocs.to_csv("portfolio_construction_allocations.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Fast portfolio optimizer sweep using existing ML predictions")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="quick is faster; full tests many more combinations")
    args = parser.parse_args()

    print("=" * 80)
    print("ALGOTRADER V7.4: FAST PORTFOLIO OPTIMIZER SWEEP + CASH YIELD")
    print("No ML retraining. Reuses existing prediction files.")
    print("=" * 80)

    scores, data = pc.load_data()
    print(f"Loaded {len(scores):,} ML prediction rows")

    all_dates = sorted(scores["date"].dropna().unique())
    rebal_dates = all_dates[::pc.HOLD_DAYS]
    start_for_cash = pd.to_datetime(min(rebal_dates)).strftime("%Y-%m-%d") if rebal_dates else None
    end_for_cash = (pd.to_datetime(max(rebal_dates)) + pd.Timedelta(days=20)).strftime("%Y-%m-%d") if rebal_dates else None
    print(f"Downloading idle-cash proxy {pc.CASH_PROXY} once...")
    cash_returns, cash_df = pc.cash_proxy_returns(start_for_cash, end_for_cash, rebal_dates)

    print("Simulating baseline once...")
    base_eq, base_trades, base_alloc = pc.simulate_baseline(scores, cash_returns)
    start = base_eq["date"].min()
    end = base_eq["date"].max()
    print("Downloading SPY once...")
    spy = pc.spy_equity(start, end)

    base_equity_only = pd.concat([base_eq, spy], ignore_index=True)
    base_overall = pc.overall_summary(base_equity_only, base_trades)
    baseline_row = base_overall[base_overall["asset"] == "ML Baseline"].iloc[0].to_dict()
    spy_row = base_overall[base_overall["asset"] == "SPY"].iloc[0].to_dict()

    configs = list(build_configs(args.mode))
    print(f"Testing {len(configs)} optimizer configurations...")

    rows = []
    best_cfg = None
    best_score = -1e9

    for i, cfg in enumerate(configs, 1):
        set_config(cfg)
        opt_eq, opt_trades, opt_alloc = pc.simulate_optimized(scores, data, cash_returns)

        common_start = max(base_eq["date"].min(), opt_eq["date"].min(), spy["date"].min() if not spy.empty else base_eq["date"].min())
        common_end = min(base_eq["date"].max(), opt_eq["date"].max(), spy["date"].max() if not spy.empty else base_eq["date"].max())
        eq = pd.concat([opt_eq, spy], ignore_index=True)
        eq = eq[(eq["date"] >= common_start) & (eq["date"] <= common_end)].copy()
        tr = opt_trades[(opt_trades["date"] >= common_start) & (opt_trades["date"] <= common_end)].copy()
        al = opt_alloc.copy()
        if not al.empty:
            al["date"] = pd.to_datetime(al["date"])
            al = al[(al["date"] >= common_start) & (al["date"] <= common_end)].copy()

        row = summarize_opt(eq, tr, al, baseline_row, spy_row)
        if row is not None:
            row.update(cfg)
            row["config_id"] = i
            row["optimizer_score"] = score_config(row)
            rows.append(row)
            if row["optimizer_score"] > best_score:
                best_score = row["optimizer_score"]
                best_cfg = cfg.copy()

        if i % 10 == 0 or i == len(configs):
            print(f"  completed {i}/{len(configs)} configs")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No optimizer results were generated.")

    results = results.sort_values("optimizer_score", ascending=False).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results.to_csv("portfolio_optimization_sweep_results.csv", index=False)

    chart_files = make_sweep_charts(results)
    make_sweep_html(results, baseline_row, spy_row, chart_files, args.mode)

    print("Saving detailed report for best configuration...")
    save_best_detail(best_cfg, scores, data, base_eq, base_trades, base_alloc, spy, cash_returns)

    best = results.iloc[0]
    print("\n" + "=" * 80)
    print("BEST PRACTICAL CONFIG")
    print("=" * 80)
    print(best[[
        "cagr_pct", "max_drawdown_pct", "sharpe", "sortino", "calmar", "avg_gross_exposure_pct",
        "cagr_vs_spy", "dd_improvement_vs_baseline", "target_vol", "max_exposure", "min_exposure",
        "max_position", "max_corr", "drawdown_mode", "prob_threshold", "slippage_bps", "max_holdings"
    ]].to_string())
    print("\nSaved:")
    print("  portfolio_optimization_sweep_report.html")
    print("  portfolio_optimization_sweep_results.csv")
    print("  portfolio_construction_report.html  (best config detail)")
    for f in chart_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
