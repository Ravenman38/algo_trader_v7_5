# AlgoTrader v7 — Portfolio Construction

Version 7 keeps the ML probability model from v6 and adds a portfolio-construction layer focused on drawdown reduction and risk-adjusted returns.

## What changed in v7

The goal is no longer to improve the ML model. The goal is to build a better portfolio from the ML probabilities.

New file:

```text
run_portfolio_construction.py
```

It compares:

1. **ML Baseline** — top-probability names with simple capped allocation.
2. **ML Optimized** — probability-weighted, volatility-targeted, correlation-aware allocation.
3. **SPY** — buy-and-hold benchmark over the same dates.

The optimized portfolio includes:

- probability-weighted sizing
- volatility-adjusted sizing
- maximum position size reduced to 10%
- minimum position size control
- correlation filtering
- volatility targeting
- market regime exposure caps
- drawdown-based exposure cuts
- commissions
- slippage
- exposure and allocation diagnostics

## Colab quick start

```python
%cd /content
!rm -rf algo_trader_v7
!git clone https://github.com/Ravenman38/algo_trader_v7.git
%cd /content/algo_trader_v7
!pip install -q -r requirements.txt
!python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2
!python run.py construct
```

Open:

```text
portfolio_construction_report.html
```

## Full research run

```python
%cd /content
!rm -rf algo_trader_v7
!git clone https://github.com/Ravenman38/algo_trader_v7.git
%cd /content/algo_trader_v7
!pip install -q -r requirements.txt
!python run.py full-research --start 2016-01-01 --model gbdt --min-train-years 2
```

This generates:

```text
walkforward_comparison_report.html
validation_report.html
data_integrity_report.html
sensitivity_analysis_report.html
portfolio_construction_report.html
```

## Important

This is still research software. Do not trade real capital until paper trading confirms that live behavior matches the backtest.
