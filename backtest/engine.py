"""
Simple event-driven-ish backtest: each trading day, rank the universe,
take the top N names by composite score, hold each for a fixed number
of days, then exit. Tracks realistic-ish costs (commission + slippage)
so the result isn't fantasy.

This is intentionally simple (no overlapping-position sizing logic,
no portfolio-level risk caps) -- a first pass to see if the signal has
ANY edge before adding complexity. Don't mistake this for a
production-grade backtest engine.
"""

import pandas as pd
import numpy as np
from signals.scorer import compute_ticker_scores, rank_universe_on_date


class BacktestConfig:
    def __init__(self,
                 top_n: int = 10,
                 hold_days: int = 3,
                 commission_per_trade: float = 1.0,
                 slippage_bps: float = 5.0,  # basis points, applied each way
                 starting_capital: float = 100_000.0):
        self.top_n = top_n
        self.hold_days = hold_days
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.starting_capital = starting_capital


def run_backtest(price_data: dict[str, pd.DataFrame], config: BacktestConfig) -> pd.DataFrame:
    """
    Returns a DataFrame of individual simulated trades with entry/exit
    dates, prices, and net return after costs.
    """
    print("[backtest] Computing indicators for all tickers...")
    scores_by_ticker = {t: compute_ticker_scores(df) for t, df in price_data.items()}

    # build the common set of trading dates across the universe
    all_dates = sorted(set().union(*[df.index for df in scores_by_ticker.values()]))

    trades = []
    slip = config.slippage_bps / 10_000.0

    print(f"[backtest] Scanning {len(all_dates)} trading days...")
    for i, date in enumerate(all_dates):
        ranked = rank_universe_on_date(scores_by_ticker, date)
        if ranked.empty:
            continue

        top_picks = ranked.head(config.top_n)

        for ticker, row in top_picks.iterrows():
            df = price_data[ticker]
            if date not in df.index:
                continue
            entry_idx = df.index.get_loc(date)
            exit_idx = entry_idx + config.hold_days
            if exit_idx >= len(df):
                continue  # not enough future data to simulate the hold

            entry_price = df["Close"].iloc[entry_idx] * (1 + slip)  # buy slightly worse than close
            exit_price = df["Close"].iloc[exit_idx] * (1 - slip)   # sell slightly worse than close
            exit_date = df.index[exit_idx]

            gross_return = (exit_price - entry_price) / entry_price
            # commission as a fraction of a notional $1000 per-trade position for simplicity
            cost_drag = (2 * config.commission_per_trade) / 1000.0
            net_return = gross_return - cost_drag

            trades.append({
                "ticker": ticker,
                "entry_date": date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "composite_score": row["composite_score"],
                "gross_return": gross_return,
                "net_return": net_return,
            })

    return pd.DataFrame(trades)


def run_regime_switch_backtest(price_data: dict[str, pd.DataFrame],
                                 benchmark_df: pd.DataFrame,
                                 config: BacktestConfig) -> pd.DataFrame:
    """
    Same mechanics as run_backtest, but picks which scoring strategy to
    use each day based on the benchmark's regime:
      - "trending" regime -> momentum strategy (compute_momentum_scores)
      - "choppy" regime    -> accumulation-footprint strategy (compute_ticker_scores)

    Tracks open positions per ticker so we don't re-enter a stock while
    already holding it -- this keeps trades genuinely independent, which
    is required for the significance test to mean anything, especially
    at longer hold periods like 21 days where daily re-entry would create
    massively overlapping, correlated trades.
    """
    from signals.regime import classify_regime
    from signals.momentum_strategy import compute_momentum_scores, rank_momentum_on_date

    print("[regime-backtest] Classifying market regime from benchmark...")
    regime_series = classify_regime(benchmark_df)
    if regime_series.index.tz is not None:
        regime_series.index = regime_series.index.tz_localize(None)
    regime_series.index = regime_series.index.normalize()

    print("[regime-backtest] Computing accumulation-footprint scores for all tickers...")
    normalized_data = {}
    for t, df in price_data.items():
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        normalized_data[t] = df
    price_data = normalized_data

    accum_scores = {t: compute_ticker_scores(df) for t, df in price_data.items()}

    print("[regime-backtest] Computing momentum scores for all tickers...")
    momentum_scores = {t: compute_momentum_scores(df) for t, df in price_data.items()}

    all_dates = sorted(set().union(*[df.index for df in accum_scores.values()]))
    trades = []
    slip = config.slippage_bps / 10_000.0

    # Track open positions: {ticker: exit_date}
    # Skip re-entry for any ticker already held, until its exit date passes.
    # This ensures each trade is a genuinely independent observation.
    open_positions = {}

    print(f"[regime-backtest] Scanning {len(all_dates)} trading days...")
    for date in all_dates:
        # Close any positions whose exit date has arrived or passed
        open_positions = {t: ex for t, ex in open_positions.items() if ex > date}

        if date not in regime_series.index:
            continue
        regime = regime_series.loc[date]

        if regime == "trending":
            ranked = rank_momentum_on_date(momentum_scores, date)
        else:
            ranked = rank_universe_on_date(accum_scores, date)

        if ranked.empty:
            continue

        # Filter out any tickers already in an open position
        eligible = ranked[~ranked.index.isin(open_positions.keys())]
        top_picks = eligible.head(config.top_n)

        for ticker, row in top_picks.iterrows():
            df = price_data[ticker]
            if date not in df.index:
                continue
            entry_idx = df.index.get_loc(date)
            exit_idx = entry_idx + config.hold_days
            if exit_idx >= len(df):
                continue

            entry_price = df["Close"].iloc[entry_idx] * (1 + slip)
            exit_price = df["Close"].iloc[exit_idx] * (1 - slip)
            exit_date = df.index[exit_idx]

            gross_return = (exit_price - entry_price) / entry_price
            cost_drag = (2 * config.commission_per_trade) / 1000.0
            net_return = gross_return - cost_drag

            # Mark this ticker as held until exit_date
            open_positions[ticker] = exit_date

            trades.append({
                "ticker": ticker,
                "regime": regime,
                "strategy_used": "momentum" if regime == "trending" else "accumulation",
                "entry_date": date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "composite_score": row["composite_score"],
                "gross_return": gross_return,
                "net_return": net_return,
            })

    return pd.DataFrame(trades)


def summarize_results(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"error": "No trades generated -- check data availability and filters."}

    win_rate = (trades["net_return"] > 0).mean()
    avg_return = trades["net_return"].mean()
    avg_win = trades.loc[trades["net_return"] > 0, "net_return"].mean()
    avg_loss = trades.loc[trades["net_return"] <= 0, "net_return"].mean()
    total_trades = len(trades)

    # rough annualization assuming trades compound sequentially in a single slot;
    # this is a simplification -- real portfolio-level compounding depends on
    # how many positions you hold concurrently, which this simple engine doesn't model.
    cumulative = (1 + trades.sort_values("entry_date")["net_return"]).cumprod()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) else 0.0

    # Statistical significance: is avg_return distinguishable from zero,
    # given the spread of returns? One-sample t-test against 0.
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(trades["net_return"], 0.0)

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "avg_return_per_trade": round(avg_return, 5),
        "avg_win": round(avg_win, 5) if not np.isnan(avg_win) else None,
        "avg_loss": round(avg_loss, 5) if not np.isnan(avg_loss) else None,
        "t_stat": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "significant_at_5pct": bool(p_value < 0.05),
        "total_compounded_return_if_sequential": round(total_return, 4),
        "note": "total_compounded_return assumes trades compound one after another in a single "
                "capital slot, NOT a realistic multi-position portfolio. Use win_rate and "
                "avg_return_per_trade as the primary signal-quality metrics, not the compounded number. "
                "p_value < 0.05 is a loose rule of thumb for 'distinguishable from zero', not proof "
                "of a real tradeable edge -- still subject to overfitting and regime-dependence.",
    }


def random_baseline(price_data: dict[str, pd.DataFrame], config, n_trials: int = 5, seed: int = 0) -> dict:
    """
    Comparison baseline: instead of ranking by composite score, pick
    top_n RANDOM tickers each day and simulate the same hold period.
    Averaged over n_trials random seeds to reduce noise in the baseline
    itself.

    If the real strategy's avg_return_per_trade isn't meaningfully better
    than this random baseline, the indicators aren't adding value --
    you're just capturing the general drift of the universe you selected.
    """
    import random as _random
    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    slip = config.slippage_bps / 10_000.0

    trial_avg_returns = []
    for trial in range(n_trials):
        _random.seed(seed + trial)
        returns = []
        for date in all_dates:
            tickers_with_data = [t for t, df in price_data.items() if date in df.index]
            if len(tickers_with_data) < config.top_n:
                continue
            picks = _random.sample(tickers_with_data, config.top_n)
            for ticker in picks:
                df = price_data[ticker]
                entry_idx = df.index.get_loc(date)
                exit_idx = entry_idx + config.hold_days
                if exit_idx >= len(df):
                    continue
                entry_price = df["Close"].iloc[entry_idx] * (1 + slip)
                exit_price = df["Close"].iloc[exit_idx] * (1 - slip)
                gross_return = (exit_price - entry_price) / entry_price
                cost_drag = (2 * config.commission_per_trade) / 1000.0
                returns.append(gross_return - cost_drag)
        if returns:
            trial_avg_returns.append(np.mean(returns))

    return {
        "random_baseline_avg_return_per_trade": round(np.mean(trial_avg_returns), 5) if trial_avg_returns else None,
        "random_baseline_trials": n_trials,
        "note": "If the strategy's avg_return_per_trade isn't clearly above this baseline, "
                "the composite score isn't adding value over picking names at random from "
                "the same filtered universe -- the edge is likely just the universe's general drift.",
    }
