"""
Indicator-based "accumulation footprint" signals.

These are all derived purely from OHLCV data -- no SEC filings, no lag.
The tradeoff (be honest with yourself about this when reading results):
a volume spike or OBV trend can be caused by lots of things besides
institutional accumulation -- index rebalancing, news, retail momentum,
options expiry effects. These are proxies, not direct observations.
"""

import numpy as np
import pandas as pd


def relative_volume(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Today's volume vs trailing average volume. >1.5-2x is generally
    considered 'unusual'.
    """
    avg_vol = df["Volume"].rolling(lookback).mean()
    return df["Volume"] / avg_vol


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """
    Classic OBV: running total that adds volume on up days, subtracts on
    down days. A rising OBV while price is flat/consolidating can suggest
    accumulation happening before price confirms it.
    """
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def obv_trend_score(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Normalized slope of OBV over the lookback window. Positive and large
    means strong sustained buying pressure relative to recent history.
    """
    obv = on_balance_volume(df)
    obv_change = obv.diff(lookback)
    # normalize by average volume over the period so the score is comparable
    # across stocks of very different sizes
    avg_vol = df["Volume"].rolling(lookback).mean()
    return obv_change / (avg_vol * lookback)


def vwap_deviation(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    How far current price sits above/below a rolling volume-weighted
    average price. Sustained positive deviation on rising volume can
    suggest persistent buying pressure (institutional orders often work
    around VWAP).
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical_price * df["Volume"]).rolling(lookback).sum() / df["Volume"].rolling(lookback).sum()
    return (df["Close"] - vwap) / vwap


def consolidation_breakout_score(df: pd.DataFrame, range_window: int = 20, breakout_window: int = 5) -> pd.Series:
    """
    Flags a breakout from a tight price range on rising volume -- the
    "someone large just stepped in" pattern. Score is higher when:
    (a) the prior range was unusually tight (low volatility), and
    (b) the recent move out of that range came with above-average volume.
    """
    rolling_high = df["High"].rolling(range_window).max()
    rolling_low = df["Low"].rolling(range_window).min()
    range_tightness = (rolling_high - rolling_low) / df["Close"]  # smaller = tighter range
    # invert so tighter ranges score higher
    tightness_score = 1 / (range_tightness + 1e-6)
    tightness_score = (tightness_score - tightness_score.rolling(252, min_periods=20).mean()) / \
                       (tightness_score.rolling(252, min_periods=20).std() + 1e-9)

    breakout = df["Close"] > rolling_high.shift(1)
    vol_confirm = relative_volume(df, range_window) > 1.3

    raw_score = tightness_score.clip(-3, 3) * breakout.astype(int) * vol_confirm.astype(int)
    return raw_score


def trend_filter(df: pd.DataFrame, ma_window: int = 100) -> pd.Series:
    """
    Simple structural filter: only consider names not in a clear downtrend.
    Returns a boolean series.
    """
    ma = df["Close"].rolling(ma_window).mean()
    return df["Close"] > ma
