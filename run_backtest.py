"""
Run this to execute a full backtest:

    python run_backtest.py

Requires internet access -- run on your own machine or Google Colab.

Two-stage screening approach:
  Stage 1: string pre-filter removes warrants, units, rights etc (instant)
  Stage 2: lightweight 15-month download screens for quality + momentum
  Full history downloaded ONLY for tickers that pass both stages.
"""

import sys, os, random, datetime, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.provider import YFinanceProvider
from data.universe import get_universe, get_market_caps
from data.screener import prefilter_tickers, parallel_screen
from backtest.dynamic_backtest import (
    run_dynamic_backtest, summarize_dynamic_results, DynamicBacktestConfig
)

# ---- Universe ----
UNIVERSE_MODE  = "broad"
MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 2_000_000_000
STARTING_CAPITAL = 100_000.0

# ---- Date range ----
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

# ---- Strategy config ----
MOMENTUM_PCT_THRESHOLD = 0.20
TARGET_RISK_PCT        = 0.005
MAX_PYRAMIDS           = 2
COOLDOWN_DAYS          = 60
# -----------------------------------------------------------------------


def main():
    print("=" * 60)
    print("DYNAMIC MOMENTUM STRATEGY (two-stage universe screen)")
    print("=" * 60)

    print(f"\n[1/4] Building and pre-filtering universe...")
    provider = YFinanceProvider()
    tickers  = get_universe(include_midcap=True, mode=UNIVERSE_MODE)
    print(f"  -> {len(tickers)} raw tickers")

    # Stage 1: string pre-filter
    candidates = prefilter_tickers(tickers)

    # Stage 2: lightweight 15-month screen
    print(f"\n[2/4] Stage 2 screen: downloading 15 months for {len(candidates)} candidates...")
    print("  (checking liquidity, MA slope, 52w high, recent momentum)")
    screened = []
    for i, t in enumerate(candidates):
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(candidates)}] {len(screened)} passing so far...")
        df = quick_screen_ticker(t, provider)
        if df is not None:
            screened.append(t)
    print(f"  -> {len(screened)} tickers passed quick screen")

    # Market cap filter
    print(f"\n[3/4] Fetching market caps for {len(screened)} screened tickers...")
    market_caps = get_market_caps(screened)
    screened    = [
        t for t in screened
        if MIN_MARKET_CAP <= market_caps.get(t, 0) <= MAX_MARKET_CAP
    ]
    print(f"  -> {len(screened)} tickers in ${MIN_MARKET_CAP/1e9:.1f}B-"
          f"${MAX_MARKET_CAP/1e9:.1f}B range")

    if len(screened) < 5:
        print("  ERROR: universe too small. Check your network or widen filters.")
        return

    # Full history download -- only for qualified tickers
    print(f"\n[4/4] Downloading full history for {len(screened)} qualified tickers...")
    start_dt    = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    fetch_start = (start_dt - datetime.timedelta(days=490)).strftime("%Y-%m-%d")
    price_data  = provider.get_history_bulk(screened, start=fetch_start, end=END_DATE)
    price_data  = {t: df for t, df in price_data.items() if not df.empty}
    print(f"  -> {len(price_data)} tickers with full history")

    print("\nFetching SPY benchmark and running backtest...")
    spy_df = provider.get_history("SPY", start=fetch_start, end=END_DATE)

    config = DynamicBacktestConfig(
        momentum_pct_threshold = MOMENTUM_PCT_THRESHOLD,
        target_risk_pct        = TARGET_RISK_PCT,
        max_pyramids           = MAX_PYRAMIDS,
        cooldown_days          = COOLDOWN_DAYS,
        slippage_bps           = 10.0,
        commission_per_trade   = 1.0,
    )
    trades_df, portfolio_df = run_dynamic_backtest(
        price_data, spy_df, spy_df, config,
        market_caps      = market_caps,
        starting_capital = STARTING_CAPITAL,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    summary = summarize_dynamic_results(
        trades_df, portfolio_df, spy_df, STARTING_CAPITAL
    )
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if not trades_df.empty:
        print("\n--- Sample trades (first 20) ---")
        cols = ["ticker", "cap_tier", "entry_date", "exit_date",
                "hold_days", "pyramids_done", "exit_reason", "net_return"]
        print(trades_df[cols].head(20).to_string(index=False))
        trades_df.to_csv("dynamic_trades.csv", index=False)
        portfolio_df.to_csv("portfolio_equity.csv")
        print("\nSaved: dynamic_trades.csv, portfolio_equity.csv")


if __name__ == "__main__":
    main()
