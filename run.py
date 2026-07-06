#!/usr/bin/env python3
"""
AlgoTrader v7.4 unified command runner.

Examples:
  python run.py screener        # rank today's candidates
  python run.py orders          # generate today's orders
  python run.py portfolio       # run portfolio backtest
  python run.py benchmark       # compare last portfolio backtest to SPY
  python run.py period          # 2018+ strategy vs SPY regime/year analysis
  python run.py train-ml        # train ML probability model
  python run.py compare-models  # compare heuristic vs ML model visually
  python run.py walkforward     # aligned ML vs heuristic vs SPY walk-forward report
  python run.py data-check      # data integrity checks
  python run.py sensitivity     # parameter robustness checks
  python run.py validate        # validation suite
  python run.py full-validate   # train 2016+ ML, walk-forward, data checks, sensitivity, validation
  python run.py construct       # portfolio construction optimizer
  python run.py optimize-portfolio # fast optimizer parameter sweep
  python run.py report          # run 2018+ heuristic period report
  python run.py all             # screener + orders + 2018+ report
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run_cmd(args: list[str]) -> int:
    print("\n" + "=" * 78)
    print("RUN:", " ".join(args))
    print("=" * 78)
    return subprocess.call(args, cwd=str(ROOT))


def require_file(path: str, message: str) -> bool:
    if not (ROOT / path).exists():
        print(f"\nMissing required file: {path}")
        print(message)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="AlgoTrader v7.4 unified runner")
    parser.add_argument(
        "command",
        choices=[
            "screener",
            "orders",
            "portfolio",
            "benchmark",
            "period",
            "report",
            "train-ml",
            "compare-models",
            "walkforward",
            "ml-period",
            "data-check",
            "sensitivity",
            "validate",
            "full-validate",
            "construct",
            "optimize-portfolio",
            "universe-audit",
            "full-research",
            "all",
        ],
    )
    parser.add_argument("--start", default="2016-01-01", help="ML training start date")
    parser.add_argument("--model", default="gbdt", choices=["gbdt", "logistic"], help="ML model type")
    parser.add_argument("--min-train-years", type=int, default=2, help="Initial walk-forward training years; default 2 for 2016+ data")
    parser.add_argument("--max-tickers", type=int, default=900, help="Maximum tickers to download/train on")
    parser.add_argument("--universe", choices=["sp", "broad"], default="sp", help="sp=current S&P500+400; broad=NASDAQ/NYSE listed names")
    parser.add_argument("--cap-mode", choices=["none", "proxy"], default="none", help="proxy applies historical market-cap proxy filter")
    parser.add_argument("--min-market-cap", type=float, default=300_000_000)
    parser.add_argument("--max-market-cap", type=float, default=10_000_000_000)
    parser.add_argument("--cache-dir", default="data_cache", help="cache downloaded price/cap data; use a Google Drive path to persist across runtime restarts")
    parser.add_argument("--refresh-data", action="store_true", help="force fresh download instead of using cache")
    args = parser.parse_args()

    if args.command == "screener":
        return run_cmd([PY, "screener_5pct.py"])

    if args.command == "orders":
        if not require_file("screener_results.csv", "Run: python run.py screener"):
            return 1
        return run_cmd([PY, "generate_orders.py"])

    if args.command == "portfolio":
        return run_cmd([PY, "run_portfolio_backtest.py"])

    if args.command == "benchmark":
        if not require_file("portfolio_backtest_equity.csv", "Run: python run.py portfolio"):
            return 1
        return run_cmd([PY, "run_benchmark_report.py"])

    if args.command in {"period", "report"}:
        return run_cmd([PY, "run_2018_period_analysis.py"])

    if args.command == "train-ml":
        return run_cmd([
            PY, "train_probability_model.py",
            "--start", args.start,
            "--model", args.model,
            "--min-train-years", str(args.min_train_years),
            "--max-tickers", str(args.max_tickers),
            "--universe", args.universe,
            "--cap-mode", args.cap_mode,
            "--min-market-cap", str(args.min_market_cap),
            "--max-market-cap", str(args.max_market_cap),
            "--cache-dir", args.cache_dir,
        ] + (["--refresh-data"] if args.refresh_data else []))

    if args.command == "compare-models":
        ok = True
        ok &= require_file("probability_training_dataset.csv", "Run: python run.py train-ml")
        ok &= require_file("ml_probability_predictions.csv", "Run: python run.py train-ml")
        if not ok:
            return 1
        return run_cmd([PY, "run_model_comparison.py"])

    if args.command in {"walkforward", "ml-period"}:
        ok = True
        ok &= require_file("probability_training_dataset.csv", "Run: python run.py train-ml")
        ok &= require_file("ml_probability_predictions.csv", "Run: python run.py train-ml")
        if not ok:
            return 1
        return run_cmd([PY, "run_walkforward_comparison.py"])


    if args.command == "data-check":
        if not require_file("probability_training_dataset.csv", "Run: python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2"):
            return 1
        return run_cmd([PY, "run_data_integrity_checks.py"])

    if args.command == "sensitivity":
        ok = True
        ok &= require_file("probability_training_dataset.csv", "Run: python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2")
        ok &= require_file("ml_probability_predictions.csv", "Run: python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2")
        if not ok:
            return 1
        return run_cmd([PY, "run_sensitivity_analysis.py"])

    if args.command == "validate":
        ok = True
        for f in [
            "walkforward_comparison_overall.csv",
            "walkforward_comparison_yearly.csv",
            "walkforward_comparison_regimes.csv",
            "walkforward_comparison_equity.csv",
            "walkforward_comparison_trades.csv",
            "probability_model_summary.csv",
            "probability_model_feature_importance.csv",
        ]:
            ok &= require_file(f, "Run: python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2 && python run.py walkforward")
        if not ok:
            return 1
        return run_cmd([PY, "run_validation_suite.py"])

    if args.command == "full-validate":
        steps = [
            [PY, "train_probability_model.py", "--start", args.start, "--model", args.model, "--min-train-years", str(args.min_train_years), "--max-tickers", str(args.max_tickers), "--universe", args.universe, "--cap-mode", args.cap_mode, "--min-market-cap", str(args.min_market_cap), "--max-market-cap", str(args.max_market_cap), "--cache-dir", args.cache_dir] + (["--refresh-data"] if args.refresh_data else []),
            [PY, "run_walkforward_comparison.py"],
            [PY, "run_data_integrity_checks.py"],
            [PY, "run_sensitivity_analysis.py"],
            [PY, "run_validation_suite.py"],
        ]
        for step in steps:
            rc = run_cmd(step)
            if rc != 0:
                return rc
        print("\nDone. Open validation_report.html, sensitivity_analysis_report.html, data_integrity_report.html, and walkforward_comparison_report.html.")
        return 0


    if args.command == "construct":
        ok = True
        ok &= require_file("probability_training_dataset.csv", "Run: python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2")
        ok &= require_file("ml_probability_predictions.csv", "Run: python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2")
        if not ok:
            return 1
        return run_cmd([PY, "run_portfolio_construction.py"])


    if args.command == "optimize-portfolio":
        ok = True
        ok &= require_file("probability_training_dataset.csv", "Run once first: python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2")
        ok &= require_file("ml_probability_predictions.csv", "Run once first: python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2")
        if not ok:
            return 1
        return run_cmd([PY, "run_portfolio_optimization_sweep.py", "--mode", "quick"])


    if args.command == "universe-audit":
        ok = True
        ok &= require_file("probability_training_dataset.csv", "Run: python run.py train-ml --cap-mode proxy")
        ok &= require_file("ml_probability_predictions.csv", "Run: python run.py train-ml --cap-mode proxy")
        if not ok:
            return 1
        return run_cmd([PY, "run_universe_audit.py"])

    if args.command == "full-research":
        steps = [
            [PY, "train_probability_model.py", "--start", args.start, "--model", args.model, "--min-train-years", str(args.min_train_years), "--max-tickers", str(args.max_tickers), "--universe", args.universe, "--cap-mode", args.cap_mode, "--min-market-cap", str(args.min_market_cap), "--max-market-cap", str(args.max_market_cap), "--cache-dir", args.cache_dir] + (["--refresh-data"] if args.refresh_data else []),
            [PY, "run_walkforward_comparison.py"],
            [PY, "run_data_integrity_checks.py"],
            [PY, "run_sensitivity_analysis.py"],
            [PY, "run_validation_suite.py"],
            [PY, "run_portfolio_construction.py"],
        ]
        for step in steps:
            rc = run_cmd(step)
            if rc != 0:
                return rc
        print("\nDone. Open portfolio_construction_report.html and validation_report.html.")
        return 0

    if args.command == "all":
        steps = [
            [PY, "screener_5pct.py"],
            [PY, "generate_orders.py"],
            [PY, "run_2018_period_analysis.py"],
        ]
        for step in steps:
            rc = run_cmd(step)
            if rc != 0:
                return rc
        print("\nDone. Open period_analysis_report.html for the full visual report.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
