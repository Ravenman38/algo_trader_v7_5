# AlgoTrader v7.4 — Original Regime Filter + Idle Cash Yield

This version keeps the v7.2 historical universe / market-cap proxy improvement, reverts to the original simple SPY regime filter, and adds idle-cash yield.

## What changed

- Uses the original/simple regime filter:
  - SPY above/below 200-day moving average
  - SPY 21-day volatility thresholds
- Parks idle cash in a cash proxy instead of assuming idle cash earns 0%.
- Default cash proxy: `SGOV`
- Fallback cash yield if SGOV download is unavailable: 4% annualized
- Portfolio reports now include idle-cash exposure and cash P&L.

## Colab command

Upload this repo to GitHub as `algo_trader_v7_4`, then run:

```python
%cd /content
!rm -rf algo_trader_v7_4
!git clone https://github.com/Ravenman38/algo_trader_v7_4.git
%cd /content/algo_trader_v7_4
!pip install -q -r requirements.txt
!python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2 --universe broad --cap-mode proxy --min-market-cap 300000000 --max-market-cap 10000000000 --max-tickers 1200
!python run.py universe-audit
!python run.py walkforward
!python run.py construct
```

## Files to upload back to ChatGPT

- `portfolio_construction_report.html`
- `walkforward_comparison_report.html`
- `universe_audit_report.html`

## Notes

This is still a research backtest. The market-cap filter is a free-data proxy, not a paid institutional point-in-time database.

## Data cache added

Version 7.4 cached saves downloaded Yahoo price history and market-cap lookup data so repeat training runs do not download the full universe again.

Default local cache:

```bash
--cache-dir data_cache
```

For Colab runtime restarts, use Google Drive so the cache survives:

```bash
--cache-dir /content/drive/MyDrive/algo_trader_cache
```

Force a fresh download only when needed:

```bash
--refresh-data
```
