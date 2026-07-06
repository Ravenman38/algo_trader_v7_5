"""
Combines individual indicators into a single composite "accumulation
footprint" score per ticker, on a given date, then ranks the universe.
"""

import pandas as pd
from signals import indicators as ind


# Weights are a starting guess, not a backtested optimum. Treat them as a
# hyperparameter to sensitivity-test, not as something to trust blindly.
DEFAULT_WEIGHTS = {
    "relative_volume": 0.25,
    "obv_trend": 0.30,
    "vwap_deviation": 0.20,
    "breakout": 0.25,
}


def compute_ticker_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given one ticker's OHLCV history, compute all indicator series and
    a combined composite score, aligned by date.
    """
    out = pd.DataFrame(index=df.index)
    out["rel_vol"] = ind.relative_volume(df)
    out["obv_trend"] = ind.obv_trend_score(df)
    out["vwap_dev"] = ind.vwap_deviation(df)
    out["breakout"] = ind.consolidation_breakout_score(df)
    out["above_trend_ma"] = ind.trend_filter(df)
    out["close"] = df["Close"]
    return out


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)


def rank_universe_on_date(scores_by_ticker: dict[str, pd.DataFrame], date, weights: dict = None) -> pd.DataFrame:
    """
    Cross-sectional ranking: for a single date, pull each ticker's raw
    indicator values, z-score them across the universe (so they're
    comparable), combine into a composite, and rank.

    Only tickers passing the trend filter (above_trend_ma) are eligible --
    this matches the "don't buy accumulation in a structurally broken
    stock" filter discussed earlier.
    """
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    for ticker, df in scores_by_ticker.items():
        if date not in df.index:
            continue
        row = df.loc[date]
        if pd.isna(row[["rel_vol", "obv_trend", "vwap_dev", "breakout"]]).any():
            continue
        rows.append({
            "ticker": ticker,
            "rel_vol": row["rel_vol"],
            "obv_trend": row["obv_trend"],
            "vwap_dev": row["vwap_dev"],
            "breakout": row["breakout"],
            "above_trend_ma": row["above_trend_ma"],
            "close": row["close"],
        })

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).set_index("ticker")

    # cross-sectional z-score each factor across the universe on this date
    for col in ["rel_vol", "obv_trend", "vwap_dev", "breakout"]:
        panel[f"z_{col}"] = _zscore(panel[col])

    panel["composite_score"] = (
        weights["relative_volume"] * panel["z_rel_vol"] +
        weights["obv_trend"] * panel["z_obv_trend"] +
        weights["vwap_deviation"] * panel["z_vwap_dev"] +
        weights["breakout"] * panel["z_breakout"]
    )

    # apply trend filter as a hard gate, not just another weighted factor
    eligible = panel[panel["above_trend_ma"]].copy()
    eligible = eligible.sort_values("composite_score", ascending=False)
    return eligible[["composite_score", "rel_vol", "obv_trend", "vwap_dev", "breakout", "close"]]
