"""
A second, simpler strategy: trailing-return momentum ranking.

This is the well-documented academic-style momentum signal (rank by
trailing N-day return), intended to be active specifically during
"trending" regimes, where momentum strategies tend to do best.

Kept deliberately simple -- this is meant as a contrasting strategy for
the regime-switch experiment, not a fully-developed momentum system.
"""

import pandas as pd


def compute_momentum_scores(df: pd.DataFrame, lookback: int = 60, skip_recent: int = 5) -> pd.DataFrame:
    """
    Trailing return momentum, skipping the most recent few days to avoid
    short-term reversal noise (standard practice in academic momentum
    construction).
    """
    out = pd.DataFrame(index=df.index)
    shifted_close = df["Close"].shift(skip_recent)
    out["momentum_return"] = shifted_close.pct_change(lookback)
    out["close"] = df["Close"]

    ma = df["Close"].rolling(100).mean()
    out["above_trend_ma"] = df["Close"] > ma
    return out


def rank_momentum_on_date(scores_by_ticker: dict, date) -> pd.DataFrame:
    rows = []
    for ticker, df in scores_by_ticker.items():
        if date not in df.index:
            continue
        row = df.loc[date]
        if pd.isna(row["momentum_return"]):
            continue
        rows.append({
            "ticker": ticker,
            "composite_score": row["momentum_return"],  # reuse the same column name as the other strategy
            "above_trend_ma": row["above_trend_ma"],
            "close": row["close"],
        })
    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).set_index("ticker")
    eligible = panel[panel["above_trend_ma"]].copy()
    eligible = eligible.sort_values("composite_score", ascending=False)
    return eligible[["composite_score", "close"]]
