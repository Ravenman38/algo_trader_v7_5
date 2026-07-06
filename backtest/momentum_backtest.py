"""
Monthly-rebalance momentum backtest engine.

Key differences from the daily-rescreen engine:
- Rebalances MONTHLY (first trading day of each month), not daily.
  This matches how the academic momentum literature was constructed
  and keeps transaction costs realistic.
- Regime filter: if SPY is below its 200-day MA on rebalance day,
  take NO positions that month (go to cash). This is the single most
  important risk control for momentum strategies.
- Equal-weight across top N names.
- No re-entry during the hold period (positions are held exactly
  one month, until the next rebalance).
"""

import pandas as pd
import numpy as np
from signals.classic_momentum import rank_momentum_universe
from signals.regime import classify_regime


class MomentumBacktestConfig:
    def __init__(self,
                 top_n: int = 10,
                 slippage_bps: float = 10.0,   # slightly higher than daily -- monthly
                                                # rebalance means less frequent but
                                                # potentially larger orders
                 commission_per_trade: float = 1.0,
                 vol_adjust: bool = False):
        self.top_n = top_n
        self.slippage_bps = slippage_bps
        self.commission_per_trade = commission_per_trade
        self.vol_adjust = vol_adjust


def get_monthly_rebalance_dates(all_dates: list) -> list:
    """
    Return the first trading day of each calendar month from all_dates.
    This is when we rebalance: sell last month's holdings, buy this
    month's top-N momentum names.
    """
    dates = pd.Series(all_dates)
    monthly = dates.groupby([dates.dt.year, dates.dt.month]).first()
    return list(monthly.values)


def run_momentum_backtest(price_data: dict[str, pd.DataFrame],
                          benchmark_df: pd.DataFrame,
                          config: MomentumBacktestConfig) -> pd.DataFrame:
    """
    Simulates monthly momentum strategy with regime filter.
    Returns a DataFrame of monthly "portfolio returns" -- one row per
    rebalance period, showing what the equal-weighted top-N portfolio
    returned that month (after costs).
    """
    # Normalize timezone for all data
    def normalize_index(df):
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        return df

    price_data = {t: normalize_index(df) for t, df in price_data.items()}
    benchmark_df = normalize_index(benchmark_df)

    regime_series = classify_regime(benchmark_df)
    if regime_series.index.tz is not None:
        regime_series.index = regime_series.index.tz_localize(None)
    regime_series.index = regime_series.index.normalize()

    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    rebalance_dates = get_monthly_rebalance_dates(all_dates)

    print(f"[momentum-backtest] {len(rebalance_dates)} monthly rebalance periods")
    slip = config.slippage_bps / 10_000.0

    results = []
    for i, entry_date in enumerate(rebalance_dates[:-1]):
        exit_date = rebalance_dates[i + 1]

        # Regime check: is SPY above its 200-day MA today?
        if entry_date not in regime_series.index:
            continue
        regime = regime_series.loc[entry_date]
        if regime != "trending":
            results.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "regime": regime,
                "n_positions": 0,
                "portfolio_return": 0.0,  # in cash -- no positions taken
                "tickers_held": "",
            })
            continue

        # Rank momentum universe on this rebalance date
        ranked = rank_momentum_universe(
            price_data, entry_date,
            top_n=config.top_n,
            vol_adjust=config.vol_adjust
        )

        if ranked.empty:
            results.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "regime": regime,
                "n_positions": 0,
                "portfolio_return": 0.0,
                "tickers_held": "",
            })
            continue

        # Compute equal-weighted portfolio return for the month
        position_returns = []
        held = []
        for ticker in ranked.index:
            df = price_data[ticker]
            if entry_date not in df.index or exit_date not in df.index:
                continue
            entry_price = df.loc[entry_date, "Close"] * (1 + slip)
            exit_price = df.loc[exit_date, "Close"] * (1 - slip)
            cost_drag = (2 * config.commission_per_trade) / 1000.0
            ret = (exit_price - entry_price) / entry_price - cost_drag
            position_returns.append(ret)
            held.append(ticker)

        if not position_returns:
            continue

        portfolio_return = np.mean(position_returns)
        results.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "regime": regime,
            "n_positions": len(held),
            "portfolio_return": portfolio_return,
            "tickers_held": ", ".join(held),
        })

    return pd.DataFrame(results)


def summarize_momentum_results(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"error": "No results generated."}

    from scipy import stats

    invested = results[results["n_positions"] > 0]
    cash_months = (results["n_positions"] == 0).sum()
    total_months = len(results)

    win_rate = (invested["portfolio_return"] > 0).mean() if len(invested) else 0
    avg_monthly_return = results["portfolio_return"].mean()  # includes cash months as 0
    avg_invested_return = invested["portfolio_return"].mean() if len(invested) else 0

    # Compound all monthly returns (cash months = 0 return)
    cumulative = (1 + results["portfolio_return"]).prod() - 1
    # Annualize: (1 + total_return)^(12/n_months) - 1
    n_months = len(results)
    annualized = (1 + cumulative) ** (12 / n_months) - 1 if n_months > 0 else 0

    t_stat, p_value = stats.ttest_1samp(invested["portfolio_return"], 0.0) if len(invested) > 1 else (0, 1)

    return {
        "total_months": total_months,
        "months_invested": int(len(invested)),
        "months_in_cash_regime_filter": int(cash_months),
        "win_rate_when_invested": round(float(win_rate), 4),
        "avg_monthly_return_incl_cash": round(float(avg_monthly_return), 5),
        "avg_monthly_return_when_invested": round(float(avg_invested_return), 5),
        "total_compounded_return": round(float(cumulative), 4),
        "annualized_return_approx": round(float(annualized), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
        "significant_at_5pct": bool(p_value < 0.05),
        "note": "annualized_return_approx compounds all monthly returns including "
                "cash months (0% return). This is a single-portfolio simulation, "
                "not a multi-strategy blend.",
    }
