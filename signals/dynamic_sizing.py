"""
Dynamic position management: pyramid into winners, scale out of losers.

The core idea: position size is not fixed at entry. It evolves based on
how the trade is actually performing. Winners grow, losers shrink.

Pyramiding (adding to a winning position):
  Triggered when price moves up 1x ATR from the last add level AND
  SAR gap is still wide (trend still young) AND
  momentum is still accelerating.
  Each pyramid adds 50% of the original position size.
  Maximum 2 pyramids per trade (position can grow to 2x original).

Scaling out (reducing a losing/stalling position):
  Triggered when momentum acceleration turns negative (recent momentum
  weaker than medium-term) -- reduce by 50%.
  Triggered when SAR gap narrows below 0.5x ATR (trend aging) -- reduce
  by 25% on top of any acceleration reduction.
  Released capital goes back to the portfolio pool for redeployment.

These rules are checked daily against each open position. The market
continuously tells us where to put the money.
"""

import pandas as pd
import numpy as np


def check_pyramid_conditions(df: pd.DataFrame,
                              exit_ind: pd.DataFrame,
                              entry_ind: pd.DataFrame,
                              accel_scores: pd.Series,
                              date,
                              last_add_price: float) -> bool:
    """
    Returns True if conditions are met to add to a winning position.

    Conditions (all must be true):
      1. Price has moved up at least 1x ATR since the last add
      2. SAR gap > 1x ATR (trend still has room to run)
      3. Momentum acceleration is still positive (trend building)
    """
    if date not in df.index:
        return False

    close = df.loc[date, "Close"]
    if date not in entry_ind.index:
        return False

    atr14     = entry_ind.loc[date, "atr14"]
    sar_gap   = entry_ind.loc[date, "sar_gap_pts"]

    if pd.isna(atr14) or pd.isna(sar_gap) or atr14 <= 0:
        return False

    # Condition 1: price moved up at least 1x ATR from last add level
    price_moved_up = (close - last_add_price) >= atr14

    # Condition 2: SAR gap still wide (trend not exhausted)
    sar_gap_ok = sar_gap > (1.0 * atr14)

    # Condition 3: acceleration still positive
    if date in accel_scores.index:
        accel = accel_scores.loc[date]
        accel_ok = not pd.isna(accel) and accel > 0
    else:
        accel_ok = False

    return bool(price_moved_up and sar_gap_ok and accel_ok)


def check_scale_out_conditions(entry_ind: pd.DataFrame,
                                accel_scores: pd.Series,
                                date) -> tuple:
    """
    Returns (scale_out_fraction, reason) if position should be reduced.

    scale_out_fraction: fraction of current position to CLOSE (0 = hold,
    0.25 = close 25%, 0.50 = close 50%, 0.75 = close 75%).
    Reasons can stack: negative acceleration + narrow SAR gap = larger reduction.
    """
    if date not in entry_ind.index:
        return 0.0, None

    atr14   = entry_ind.loc[date, "atr14"]
    sar_gap = entry_ind.loc[date, "sar_gap_pts"]

    if pd.isna(atr14) or pd.isna(sar_gap) or atr14 <= 0:
        return 0.0, None

    scale_fraction = 0.0
    reasons        = []

    # Check acceleration
    if date in accel_scores.index:
        accel = accel_scores.loc[date]
        if not pd.isna(accel) and accel < 0:
            scale_fraction += 0.50
            reasons.append("decel")

    # Check SAR gap narrowing (trend aging)
    if sar_gap < (0.5 * atr14):
        scale_fraction += 0.25
        reasons.append("sar_narrow")

    # Cap at 75% reduction in a single day (always keep a residual
    # until full exit signal fires via SAR or Chandelier)
    scale_fraction = min(scale_fraction, 0.75)

    return scale_fraction, ("+".join(reasons) if reasons else None)
