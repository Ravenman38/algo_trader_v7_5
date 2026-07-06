"""
Quality filters to prevent entering structurally declining positions.

Two price-based filters that work correctly in backtests AND live trading,
requiring no paid data and no point-in-time data issues:

1. 200-day MA slope > 0:
   The long-term moving average must itself be trending upward, not just
   price being above it. A stock can bounce above its 200-day MA during a
   bear rally while the MA is still declining. Requiring the slope to be
   positive ensures the structural trend is genuinely improving.
   Computed as: MA(200) today > MA(200) 20 days ago.

2. Price within 30% of 52-week high:
   The stock must be within 30% of its highest price over the past year.
   A stock 60% below its 52-week high that bounces 25% is still deeply
   distressed -- this filters out recovering-from-collapse situations and
   keeps the universe in structural strength.
   Formula: close > 52_week_high * 0.70

Both apply equally to equities, crypto, and FX.
Cooling-off period (60 days after stop-outs) is also defined here.
"""

import pandas as pd
from datetime import timedelta


# ── 200-day MA slope filter ─────────────────────────────────────────────────

def check_ma200_slope(df: pd.DataFrame, date, slope_window: int = 20) -> bool:
    """
    Returns True if the 200-day MA is sloping upward on this date.
    Slope measured as MA[today] > MA[20 days ago].
    """
    if date not in df.index:
        return False
    ma200 = df["Close"].rolling(200).mean()
    if date not in ma200.index:
        return False
    idx = ma200.index.get_loc(date)
    if idx < slope_window:
        return False
    current_ma  = ma200.iloc[idx]
    prior_ma    = ma200.iloc[idx - slope_window]
    if pd.isna(current_ma) or pd.isna(prior_ma):
        return False
    return bool(current_ma > prior_ma)


# ── 52-week high proximity filter ───────────────────────────────────────────

def check_52week_high_proximity(df: pd.DataFrame, date,
                                 max_pct_below: float = 0.30) -> bool:
    """
    Returns True if price is within max_pct_below of the 52-week high.
    e.g. max_pct_below=0.30 means price must be > 70% of the 52-week high.

    This filters out structurally distressed stocks that are bouncing from
    a collapse -- they may show short-term momentum but are far from their
    structural strength level.
    """
    if date not in df.index:
        return False
    idx = df.index.get_loc(date)
    lookback = min(idx, 252)  # up to 52 weeks back
    if lookback < 50:
        return False
    high_52w = df["Close"].iloc[idx - lookback: idx + 1].max()
    close    = df["Close"].iloc[idx]
    if pd.isna(high_52w) or high_52w <= 0:
        return False
    return bool(close >= high_52w * (1 - max_pct_below))


# ── Pre-compute filter series for efficiency ─────────────────────────────────

def compute_quality_flags(df: pd.DataFrame,
                           slope_window: int = 20,
                           max_pct_below_52w: float = 0.30) -> pd.DataFrame:
    """
    Pre-compute both quality flags for all dates in a ticker's history.
    Returns a DataFrame with boolean columns: ma200_slope_ok, high52w_ok.
    Called once per ticker before the backtest loop for efficiency.
    """
    out = pd.DataFrame(index=df.index)

    # 200-day MA slope
    ma200            = df["Close"].rolling(200).mean()
    out["ma200_slope_ok"] = (ma200 > ma200.shift(slope_window)).fillna(False)

    # 52-week high proximity
    high_52w         = df["Close"].rolling(252).max()
    out["high52w_ok"] = (df["Close"] >= high_52w.shift(1) * (1 - max_pct_below_52w)).fillna(False)

    out["quality_ok"] = out["ma200_slope_ok"] & out["high52w_ok"]
    return out


# ── Cooling-off period tracker ───────────────────────────────────────────────

class CoolingOffTracker:
    """
    Tracks which tickers/symbols are in a cooling-off period after a
    stop-out (SAR exit or Chandelier exit).
    Works identically for equities, crypto, and FX.
    """
    def __init__(self, cooldown_days: int = 60):
        self.cooldown_days = cooldown_days
        self._cooldowns: dict[str, pd.Timestamp] = {}

    def register_stopout(self, symbol: str, exit_date):
        earliest = pd.Timestamp(exit_date) + timedelta(days=self.cooldown_days)
        existing = self._cooldowns.get(symbol)
        if existing is None or earliest > existing:
            self._cooldowns[symbol] = earliest

    def is_allowed(self, symbol: str, date) -> bool:
        deadline = self._cooldowns.get(symbol)
        if deadline is None:
            return True
        return pd.Timestamp(date) >= deadline
