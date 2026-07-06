"""
Combined strategy backtest:
  - TRENDING regime (SPY above 200-day MA): run 12-1 momentum, monthly rebalance
  - CHOPPY regime  (SPY below 200-day MA): run mean reversion, weekly rebalance

This is the full strategy combining both approaches. The two sub-strategies
are designed to be complementary:
  - Momentum profits from persistent trends
  - Mean reversion profits from short-term oversold bounces in directionless markets
  - Neither is active in the wrong regime

Key design decisions:
  - Mean reversion uses a WEEKLY rebalance (every 5 trading days) since
    mean reversion plays out much faster than momentum trends.
  - Mean reversion holds for 5 trading days maximum, or exits early if
    the stock reverts above its 20-day MA (whichever comes first).
  - Momentum rebalances MONTHLY as before.
  - Both use the same universe (liquid, market-cap filtered stocks).
  - Regime is checked on each rebalance date, not daily.
"""

import pandas as pd
import numpy as np
from signals.classic_momentum import rank_momentum_universe
from signals.mean_reversion import compute_mean_reversion_scores, rank_mean_reversion_universe
from signals.regime import classify_regime


class CombinedConfig:
    def __init__(self,
                 momentum_top_n: int = 10,
                 mr_top_n: int = 10,
                 mr_hold_days: int = 5,
                 slippage_bps: float = 10.0,
                 commission_per_trade: float = 1.0,
                 vol_adjust_momentum: bool = False,
                 mr_min_pct_below_ma20: float = -0.05,
                 mr_max_rsi: float = 35.0):
        self.momentum_top_n = momentum_top_n
        self.mr_top_n = mr_top_n
        self.mr_hold_days = mr_hold_days
        self.slippage_bps = slippage_bps
        self.commission_per_trade = commission_per_trade
        self.vol_adjust_momentum = vol_adjust_momentum
        self.mr_min_pct_below_ma20 = mr_min_pct_below_ma20
        self.mr_max_rsi = mr_max_rsi


def get_monthly_dates(all_dates: list) -> list:
    dates = pd.Series(all_dates)
    monthly = dates.groupby([dates.dt.year, dates.dt.month]).first()
    return list(monthly.values)


def get_weekly_dates(all_dates: list) -> list:
    """Every 5th trading day -- approximately weekly rebalance."""
    return all_dates[::5]


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df


def run_combined_backtest(price_data: dict[str, pd.DataFrame],
                           benchmark_df: pd.DataFrame,
                           config: CombinedConfig) -> pd.DataFrame:
    # Normalize all indexes
    price_data = {t: normalize_df(df) for t, df in price_data.items()}
    benchmark_df = normalize_df(benchmark_df)

    regime_series = classify_regime(benchmark_df)
    if regime_series.index.tz is not None:
        regime_series.index = regime_series.index.tz_localize(None)
    regime_series.index = regime_series.index.normalize()

    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    monthly_dates = get_monthly_dates(all_dates)
    weekly_dates = get_weekly_dates(all_dates)

    slip = config.slippage_bps / 10_000.0

    print("[combined] Pre-computing mean reversion scores for all tickers...")
    mr_scores = {t: compute_mean_reversion_scores(df) for t, df in price_data.items()}

    results = []

    # --- TRENDING REGIME: monthly momentum ---
    print("[combined] Running momentum sub-strategy (trending regime)...")
    for i, entry_date in enumerate(monthly_dates[:-1]):
        if entry_date not in regime_series.index:
            continue
        if regime_series.loc[entry_date] != "trending":
            continue

        exit_date = monthly_dates[i + 1]
        ranked = rank_momentum_universe(
            price_data, entry_date,
            top_n=config.momentum_top_n,
            vol_adjust=config.vol_adjust_momentum
        )
        if ranked.empty:
            continue

        position_returns = []
        held = []
        for ticker in ranked.index:
            df = price_data[ticker]
            if entry_date not in df.index or exit_date not in df.index:
                continue
            ep = df.loc[entry_date, "Close"] * (1 + slip)
            xp = df.loc[exit_date, "Close"] * (1 - slip)
            cost = (2 * config.commission_per_trade) / 1000.0
            position_returns.append((xp - ep) / ep - cost)
            held.append(ticker)

        if position_returns:
            results.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "strategy": "momentum",
                "regime": "trending",
                "n_positions": len(held),
                "portfolio_return": np.mean(position_returns),
                "tickers_held": ", ".join(held),
            })

    # --- CHOPPY REGIME: weekly mean reversion ---
    print("[combined] Running mean reversion sub-strategy (choppy regime)...")
    for i, entry_date in enumerate(weekly_dates[:-1]):
        if entry_date not in regime_series.index:
            continue
        if regime_series.loc[entry_date] != "choppy":
            continue

        # Exit after mr_hold_days trading days
        entry_idx = all_dates.index(entry_date)
        exit_idx = min(entry_idx + config.mr_hold_days, len(all_dates) - 1)
        exit_date = all_dates[exit_idx]

        ranked = rank_mean_reversion_universe(
            mr_scores, entry_date,
            top_n=config.mr_top_n,
            min_pct_below_ma20=config.mr_min_pct_below_ma20,
            max_rsi=config.mr_max_rsi
        )
        if ranked.empty:
            continue

        position_returns = []
        held = []
        for ticker in ranked.index:
            df = price_data[ticker]
            if entry_date not in df.index or exit_date not in df.index:
                continue
            ep = df.loc[entry_date, "Close"] * (1 + slip)
            xp = df.loc[exit_date, "Close"] * (1 - slip)
            cost = (2 * config.commission_per_trade) / 1000.0
            position_returns.append((xp - ep) / ep - cost)
            held.append(ticker)

        if position_returns:
            results.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "strategy": "mean_reversion",
                "regime": "choppy",
                "n_positions": len(held),
                "portfolio_return": np.mean(position_returns),
                "tickers_held": ", ".join(held),
            })

    df_results = pd.DataFrame(results).sort_values("entry_date").reset_index(drop=True)
    return df_results


def summarize_combined_results(results: pd.DataFrame,
                                benchmark_df: pd.DataFrame) -> dict:
    if results.empty:
        return {"error": "No results generated."}

    from scipy import stats

    # --- Per-strategy breakdown ---
    mom = results[results["strategy"] == "momentum"]
    mr = results[results["strategy"] == "mean_reversion"]

    # --- SPY benchmark return over same period ---
    bench = normalize_df(benchmark_df.copy())
    start = results["entry_date"].min()
    end = results["exit_date"].max()
    bench_window = bench[(bench.index >= start) & (bench.index <= end)]
    if len(bench_window) > 1:
        spy_total = (bench_window["Close"].iloc[-1] / bench_window["Close"].iloc[0]) - 1
        n_years = (bench_window.index[-1] - bench_window.index[0]).days / 365.25
        spy_annualized = (1 + spy_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    else:
        spy_total = spy_annualized = 0

    # --- Strategy compounded return ---
    # For a fair comparison we need to account for the fact that momentum
    # and mean reversion periods don't always overlap -- mean reversion
    # runs weekly sub-periods within choppy months.
    # Simplification: compound all period returns sequentially.
    sorted_results = results.sort_values("entry_date")
    cumulative = (1 + sorted_results["portfolio_return"]).prod() - 1
    n_days = (sorted_results["exit_date"].max() - sorted_results["entry_date"].min()).days
    n_years = n_days / 365.25 if n_days > 0 else 1
    annualized = (1 + cumulative) ** (1 / n_years) - 1 if n_years > 0 else 0

    excess_return = annualized - spy_annualized

    # Significance on momentum (monthly, independent)
    t_stat, p_val = stats.ttest_1samp(mom["portfolio_return"], 0.0) if len(mom) > 1 else (0, 1)
    mr_t, mr_p = stats.ttest_1samp(mr["portfolio_return"], 0.0) if len(mr) > 1 else (0, 1)

    return {
        "total_periods": len(results),
        "momentum_periods": len(mom),
        "mean_reversion_periods": len(mr),
        "momentum_win_rate": round(float((mom["portfolio_return"] > 0).mean()), 4) if len(mom) else None,
        "momentum_avg_return": round(float(mom["portfolio_return"].mean()), 5) if len(mom) else None,
        "momentum_p_value": round(float(p_val), 4),
        "mr_win_rate": round(float((mr["portfolio_return"] > 0).mean()), 4) if len(mr) else None,
        "mr_avg_return_per_period": round(float(mr["portfolio_return"].mean()), 5) if len(mr) else None,
        "mr_p_value": round(float(mr_p), 4),
        "strategy_total_compounded_return": round(float(cumulative), 4),
        "strategy_annualized_return": round(float(annualized), 4),
        "spy_total_return_same_period": round(float(spy_total), 4),
        "spy_annualized_return": round(float(spy_annualized), 4),
        "excess_return_vs_spy": round(float(excess_return), 4),
        "beating_market_by_5pct": bool(excess_return >= 0.05),
        "note": "excess_return is strategy annualized minus SPY annualized over the same "
                "dates. Mean reversion periods overlap within choppy months so compounding "
                "is approximate. A positive excess_return does not guarantee future "
                "outperformance -- validate out-of-sample before trusting.",
    }
