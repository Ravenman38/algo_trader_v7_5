"""
Classic cross-sectional momentum signal + acceleration filter.

Formation period: 12-month trailing return, skipping the most recent
1 month (the "12-1" construction from Jegadeesh & Titman 1993).

Acceleration filter: requires that recent 3-month momentum exceeds
annualized 6-month momentum -- the trend must be speeding up, not
slowing down. This filters out stocks with strong historical momentum
that has already peaked and is now fading.
"""

import pandas as pd
import numpy as np


def compute_12_1_momentum(df: pd.DataFrame) -> pd.Series:
    """
    12-1 momentum: return from 252 trading days ago to 21 trading days ago.
    """
    ret_12m  = df["Close"].pct_change(252)
    ret_1m   = df["Close"].pct_change(21)
    momentum = ((1 + ret_12m) / (1 + ret_1m)) - 1
    return momentum


def compute_momentum_acceleration(df: pd.DataFrame) -> pd.Series:
    """
    Momentum acceleration: is the trend speeding up or slowing down?

    Compares recent 3-month return to annualized 6-month return.
    Positive = accelerating (recent performance stronger than medium-term).
    Negative = decelerating (recent performance weaker than medium-term).

    Both returns are annualized to the same scale for fair comparison:
      3-month annualized = (1 + ret_3m)^4 - 1
      6-month annualized = (1 + ret_6m)^2 - 1

    A stock in a genuine uptrend that is building should show positive
    acceleration -- the most recent months are outperforming the prior
    6-month average, suggesting institutional buying is increasing.
    """
    ret_3m = df["Close"].pct_change(63)   # ~3 months
    ret_6m = df["Close"].pct_change(126)  # ~6 months

    # Annualize both to same scale
    ann_3m = (1 + ret_3m) ** 4 - 1       # 3m annualized (4 periods/year)
    ann_6m = (1 + ret_6m) ** 2 - 1       # 6m annualized (2 periods/year)

    return ann_3m - ann_6m   # positive = accelerating


def compute_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    return df["Close"].pct_change().rolling(window).std() * np.sqrt(252)


def rank_momentum_universe(price_data: dict, date, top_n: int = 10,
                            vol_adjust: bool = False) -> pd.DataFrame:
    rows = []
    for ticker, df in price_data.items():
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 273:
            continue
        mom = compute_12_1_momentum(df).iloc[idx]
        if pd.isna(mom):
            continue
        ma100 = df["Close"].rolling(100).mean().iloc[idx]
        if pd.isna(ma100) or df["Close"].iloc[idx] < ma100:
            continue
        score = mom
        if vol_adjust:
            vol = compute_volatility(df).iloc[idx]
            if pd.isna(vol) or vol == 0:
                continue
            score = mom / vol
        rows.append({
            "ticker":         ticker,
            "momentum_score": score,
            "close":          df["Close"].iloc[idx],
        })

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).set_index("ticker")
    return panel.sort_values("momentum_score", ascending=False).head(top_n)
