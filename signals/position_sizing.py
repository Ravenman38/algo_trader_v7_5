"""
Risk-based position sizing combining ATR (idiosyncratic risk) and
beta (systematic/market risk).

ATR-based sizing:
  Sets the base position size so that a 1x ATR(14) move = target_risk_pct
  of portfolio value. Lower-volatility stocks get more capital.

Beta adjustment:
  Divides the ATR-based size by the stock's rolling beta vs SPY.
  - Beta > 1: stock amplifies market moves -> smaller position
  - Beta < 1: stock dampens market moves  -> larger position
  - Beta = 1: no adjustment

Combined: position_size = (portfolio * target_risk_pct / ATR_pct) / beta

This means every position contributes equal idiosyncratic risk AND
equal systematic market risk to the portfolio. No arbitrary caps --
the risk model itself governs sizing.

Beta is computed as a rolling 63-day (3-month) regression of daily
stock returns on SPY returns. Rolling rather than static because beta
changes over time -- a stock's market sensitivity in a bull market
differs from a bear market.
"""

import pandas as pd
import numpy as np


def compute_rolling_beta(stock_df: pd.DataFrame,
                          spy_df: pd.DataFrame,
                          window: int = 63) -> pd.Series:
    """
    Rolling beta of a stock vs SPY over a given window.

    Beta = Cov(stock_returns, spy_returns) / Var(spy_returns)

    Uses a 63-day (3-month) window by default -- short enough to
    be responsive to regime changes, long enough to be stable.
    Returns NaN where insufficient history exists.
    """
    stock_ret = stock_df["Close"].pct_change()
    spy_ret   = spy_df["Close"].pct_change()

    # Align indexes
    combined  = pd.DataFrame({"stock": stock_ret, "spy": spy_ret}).dropna()

    rolling_cov = combined["stock"].rolling(window).cov(combined["spy"])
    rolling_var = combined["spy"].rolling(window).var()

    beta = rolling_cov / rolling_var
    # Re-index to match the original stock DataFrame index
    return beta.reindex(stock_df.index)


def compute_position_size(portfolio_value: float,
                           target_risk_pct: float,
                           atr14: float,
                           entry_price: float,
                           beta: float,
                           min_beta: float = 0.3,
                           max_position_pct: float = 0.20) -> float:
    """
    Compute a single position's dollar size.

    max_position_pct: hard ceiling as fraction of portfolio value.
    Default 20% -- no single position can exceed 20% of portfolio,
    regardless of how low its ATR or beta are. This prevents leverage
    blowups from near-zero ATR or near-zero beta stocks while remaining
    principled (it's a portfolio concentration limit, not an arbitrary
    dollar cap).
    """
    if atr14 <= 0 or entry_price <= 0:
        return portfolio_value * 0.05

    atr_pct = atr14 / entry_price
    if atr_pct <= 0:
        return portfolio_value * 0.05

    atr_size       = (portfolio_value * target_risk_pct) / atr_pct
    effective_beta = max(abs(beta) if not pd.isna(beta) else 1.0, min_beta)
    final_size     = atr_size / effective_beta

    # Portfolio concentration limit: principled ceiling
    return min(final_size, portfolio_value * max_position_pct)


def size_all_trades(trades: pd.DataFrame,
                    price_data: dict,
                    spy_df: pd.DataFrame,
                    starting_capital: float,
                    target_risk_pct: float = 0.01,
                    beta_window: int = 63) -> pd.DataFrame:
    """
    Compute volatility + beta adjusted position sizes for all trades.

    Returns the trades DataFrame with added columns:
      beta, atr14, position_size_dollars
    """
    from signals.entry_exit import compute_entry_indicators

    # Pre-compute rolling betas for all tickers
    print("[sizing] Computing rolling betas vs SPY...")
    betas = {}
    spy_norm = spy_df.copy()
    if spy_norm.index.tz is not None:
        spy_norm = spy_norm.copy()
        spy_norm.index = spy_norm.index.tz_localize(None)
    spy_norm.index = spy_norm.index.normalize()

    for ticker, df in price_data.items():
        try:
            betas[ticker] = compute_rolling_beta(df, spy_norm, window=beta_window)
        except Exception:
            betas[ticker] = pd.Series(1.0, index=df.index)

    results = []
    for idx, row in trades.iterrows():
        ticker      = row["ticker"]
        entry_date  = row["entry_date"]
        entry_price = row["entry_price"]

        # Get ATR14 at entry
        atr14 = None
        if ticker in price_data and entry_date in price_data[ticker].index:
            df  = price_data[ticker]
            ind = compute_entry_indicators(df, market_cap=None)
            if entry_date in ind.index and not pd.isna(ind.loc[entry_date, "atr14"]):
                atr14 = ind.loc[entry_date, "atr14"]

        # Get beta at entry
        beta = 1.0
        if ticker in betas and entry_date in betas[ticker].index:
            b = betas[ticker].loc[entry_date]
            if not pd.isna(b):
                beta = b

        if atr14 is None:
            pos_size = starting_capital * 0.05
        else:
            pos_size = compute_position_size(
                portfolio_value  = starting_capital,
                target_risk_pct  = target_risk_pct,
                atr14            = atr14,
                entry_price      = entry_price,
                beta             = beta,
            )

        results.append({
            "index":               idx,
            "beta":                round(float(beta), 3),
            "atr14":               round(float(atr14), 4) if atr14 else None,
            "position_size_dollars": round(pos_size, 2),
        })

    sizing_df = pd.DataFrame(results).set_index("index")
    return trades.join(sizing_df)
