"""
Short-term mean reversion signal for use during choppy/bear regimes.

Logic: stocks that have dropped sharply over a short window relative to
their own recent history tend to snap back. We look for:
  1. Price significantly below its 20-day moving average (oversold)
  2. RSI below 35 (momentum exhaustion)
  3. Stock still above a longer-term floor (200-day MA) to avoid
     "catching falling knives" -- stocks in genuine structural downtrends

The combination of (1) and (2) filters for genuine short-term oversold
conditions rather than secular declines.

Only deployed when the broad market regime is "choppy" (SPY below its
200-day MA), since mean reversion works poorly in trending markets where
a drop often continues rather than reverting.
"""

import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_mean_reversion_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean reversion indicators for a single ticker's OHLCV history.
    Returns a DataFrame with scores indexed by date.
    """
    out = pd.DataFrame(index=df.index)
    out["close"] = df["Close"]

    # Distance below 20-day MA -- more negative = more oversold
    ma20 = df["Close"].rolling(20).mean()
    out["pct_below_ma20"] = (df["Close"] - ma20) / (ma20 + 1e-9)

    # RSI(14) -- lower = more oversold
    out["rsi"] = compute_rsi(df["Close"], window=14)

    # Longer-term floor: is this stock in a structural downtrend?
    # If below its 200-day MA, we skip it (catching a falling knife risk)
    ma200 = df["Close"].rolling(200).mean()
    out["above_ma200"] = df["Close"] > ma200

    # 5-day return (to avoid buying into a momentum collapse)
    out["ret_5d"] = df["Close"].pct_change(5)

    # Combined oversold score: lower pct_below_ma20 (more negative) and
    # lower RSI = higher mean reversion opportunity
    # We invert and normalize so higher score = better opportunity
    out["mr_score"] = -(out["pct_below_ma20"]) * (1 - out["rsi"] / 100)

    return out


def rank_mean_reversion_universe(scores_by_ticker: dict,
                                  date,
                                  top_n: int = 10,
                                  min_pct_below_ma20: float = -0.05,
                                  max_rsi: float = 35.0) -> pd.DataFrame:
    """
    Cross-sectional ranking: find the most oversold stocks on a given date
    that pass the quality filters.

    min_pct_below_ma20: stock must be at least this far below its 20-day MA
                        (e.g., -0.05 = at least 5% below)
    max_rsi: RSI must be below this threshold (e.g., 35 = oversold)
    """
    rows = []
    for ticker, df in scores_by_ticker.items():
        if date not in df.index:
            continue
        row = df.loc[date]

        if pd.isna(row[["pct_below_ma20", "rsi", "mr_score"]]).any():
            continue

        # Quality gate: must be above 200-day MA (not in structural downtrend)
        if not row["above_ma200"]:
            continue

        # Oversold filters
        if row["pct_below_ma20"] > min_pct_below_ma20:
            continue  # not oversold enough
        if row["rsi"] > max_rsi:
            continue  # not oversold enough

        rows.append({
            "ticker": ticker,
            "mr_score": row["mr_score"],
            "pct_below_ma20": row["pct_below_ma20"],
            "rsi": row["rsi"],
            "close": row["close"],
        })

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).set_index("ticker")
    # Rank by combined oversold score, highest first
    return panel.sort_values("mr_score", ascending=False).head(top_n)
