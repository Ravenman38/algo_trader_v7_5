"""
Regime classification: is the market currently trending or choppy?

Uses a benchmark (SPY by default) and a simple, well-known rule: price
above its 200-day moving average AND the moving average itself sloping
upward = "trending". Otherwise = "choppy".

CRITICAL for avoiding lookahead bias: at any given date, this only uses
benchmark data up to and including that date. Never uses future data to
decide the regime for a past date.
"""

import pandas as pd


def classify_regime(benchmark_df: pd.DataFrame, ma_window: int = 200, slope_window: int = 20) -> pd.Series:
    """
    Returns a Series of "trending" / "choppy" indexed by date, computed
    using only data available up to that date (rolling windows naturally
    enforce this).
    """
    ma = benchmark_df["Close"].rolling(ma_window).mean()
    price_above_ma = benchmark_df["Close"] > ma
    ma_slope_positive = ma.diff(slope_window) > 0

    regime = pd.Series("choppy", index=benchmark_df.index)
    regime[price_above_ma & ma_slope_positive] = "trending"
    return regime
