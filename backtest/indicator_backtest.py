"""
Indicator-driven backtest engine with market-cap-tiered entry filters.

Entry: stocks in the top X% by momentum (tier-specific threshold)
       that also pass MACD fresh cross, MA50, volume, and SAR gap
       (all thresholds scaled to the stock's market cap tier).
Exit:  Parabolic SAR OR Chandelier 2x ATR -- whichever fires first.
Regime: only enter new positions when SPY > 200-day MA.
No arbitrary time limits.
"""

import pandas as pd
import numpy as np
from signals.classic_momentum import compute_12_1_momentum, compute_momentum_acceleration
from signals.entry_exit import (
    compute_entry_indicators, check_entry, get_momentum_threshold,
    compute_exit_indicators, find_exit_date
)
from signals.regime import classify_regime


class IndicatorBacktestConfig:
    def __init__(self,
                 momentum_pct_threshold: float = 0.20,
                 max_positions: int = 15,
                 slippage_bps: float = 10.0,
                 commission_per_trade: float = 1.0):
        self.momentum_pct_threshold = momentum_pct_threshold
        self.max_positions = max_positions
        self.slippage_bps = slippage_bps
        self.commission_per_trade = commission_per_trade


def normalize_df(df):
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df


def run_indicator_backtest(price_data, benchmark_df, config,
                            market_caps: dict = None):
    """
    market_caps: {ticker: float} -- market cap for each ticker.
    Used to determine cap tier for tier-specific entry filters.
    If None, all tickers default to "mid" tier.
    """
    price_data   = {t: normalize_df(df) for t, df in price_data.items()}
    benchmark_df = normalize_df(benchmark_df)

    regime_series = classify_regime(benchmark_df)
    if regime_series.index.tz is not None:
        regime_series.index = regime_series.index.tz_localize(None)
    regime_series.index = regime_series.index.normalize()

    market_caps = market_caps or {}

    print("[indicator-bt] Pre-computing entry indicators (cap-tiered)...")
    entry_ind = {
        t: compute_entry_indicators(df, market_cap=market_caps.get(t))
        for t, df in price_data.items()
    }

    print("[indicator-bt] Pre-computing exit indicators (SAR + Chandelier 2x ATR)...")
    exit_ind = {t: compute_exit_indicators(df) for t, df in price_data.items()}

    print("[indicator-bt] Pre-computing momentum scores and acceleration...")
    momentum_scores = {t: compute_12_1_momentum(df) for t, df in price_data.items()}
    accel_scores    = {t: compute_momentum_acceleration(df) for t, df in price_data.items()}

    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    slip      = config.slippage_bps / 10_000.0

    open_positions  = {}
    scheduled_exits = {}
    trades          = []

    print(f"[indicator-bt] Scanning {len(all_dates)} trading days...")
    for date in all_dates:

        # --- Process exits due today ---
        to_close = [t for t, (ex_date, _, _) in scheduled_exits.items()
                    if ex_date <= date]
        for ticker in to_close:
            ex_date, reason, ex_price = scheduled_exits.pop(ticker)
            pos = open_positions.pop(ticker)
            xp      = ex_price * (1 - slip)
            cost    = (2 * config.commission_per_trade) / 1000.0
            net_ret = (xp - pos["entry_price"]) / pos["entry_price"] - cost
            trades.append({
                "ticker":      ticker,
                "cap_tier":    pos.get("cap_tier", "unknown"),
                "entry_date":  pos["entry_date"],
                "exit_date":   ex_date,
                "entry_price": pos["entry_price"],
                "exit_price":  xp,
                "exit_reason": reason,
                "net_return":  net_ret,
                "hold_days":   (ex_date - pos["entry_date"]).days,
            })

        # --- Regime check ---
        if date not in regime_series.index:
            continue
        if regime_series.loc[date] != "trending":
            continue

        # No fixed position cap -- filters decide how many we hold

        # --- Compute cross-sectional momentum scores ---
        mom_scores = {}
        for t, scores in momentum_scores.items():
            if t in open_positions:
                continue
            if date not in scores.index:
                continue
            val = scores.loc[date]
            if not pd.isna(val):
                mom_scores[t] = val

        if not mom_scores:
            continue

        all_score_vals = list(mom_scores.values())

        # --- Apply entry filter with tier-specific momentum threshold ---
        slots = config.max_positions  # effectively unlimited
        entered   = 0

        # Sort all candidates by momentum score descending
        candidates = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)

        for ticker, mom_score in candidates:
            if entered >= slots:
                break

            # Acceleration filter: recent 3-month momentum must exceed
            # annualized 6-month momentum -- trend must be speeding up
            if ticker in accel_scores and date in accel_scores[ticker].index:
                accel = accel_scores[ticker].loc[date]
                if pd.isna(accel) or accel <= 0:
                    continue  # trend is decelerating -- skip

            # Get this ticker's tier-specific momentum threshold
            eff_threshold = get_momentum_threshold(
                entry_ind[ticker], date, config.momentum_pct_threshold
            )
            # Check if this score clears the tier-specific percentile
            threshold_val = np.percentile(all_score_vals,
                                          (1 - eff_threshold) * 100)
            if mom_score < threshold_val:
                continue  # doesn't pass this tier's momentum bar

            if not check_entry(entry_ind[ticker], date):
                continue

            df = price_data[ticker]
            if date not in df.index:
                continue

            entry_price = df.loc[date, "Close"] * (1 + slip)
            ex_date, reason, ex_price = find_exit_date(
                exit_ind[ticker], df, date, entry_price
            )
            cap_tier = entry_ind[ticker].loc[date, "cap_tier"] if date in entry_ind[ticker].index else "unknown"
            accel_val = accel_scores[ticker].loc[date] if ticker in accel_scores and date in accel_scores[ticker].index else None
            open_positions[ticker]  = {
                "entry_date":  date,
                "entry_price": entry_price,
                "cap_tier":    cap_tier,
            }
            scheduled_exits[ticker] = (ex_date, reason, ex_price)
            entered += 1

    # Close remaining at end of data
    last_date = all_dates[-1]
    for ticker, pos in open_positions.items():
        df = price_data[ticker]
        if last_date in df.index:
            xp      = df.loc[last_date, "Close"] * (1 - slip)
            cost    = (2 * config.commission_per_trade) / 1000.0
            net_ret = (xp - pos["entry_price"]) / pos["entry_price"] - cost
            trades.append({
                "ticker":      ticker,
                "cap_tier":    pos.get("cap_tier", "unknown"),
                "entry_date":  pos["entry_date"],
                "exit_date":   last_date,
                "entry_price": pos["entry_price"],
                "exit_price":  xp,
                "exit_reason": "end_of_backtest",
                "net_return":  net_ret,
                "hold_days":   (last_date - pos["entry_date"]).days,
            })

    return pd.DataFrame(trades)


def summarize_indicator_results(trades, benchmark_df):
    if trades.empty:
        return {"error": "No trades generated."}

    from scipy import stats

    bench        = normalize_df(benchmark_df.copy())
    start        = trades["entry_date"].min()
    end          = trades["exit_date"].max()
    bench_window = bench[(bench.index >= start) & (bench.index <= end)]

    if len(bench_window) > 1:
        spy_total = (bench_window["Close"].iloc[-1] /
                     bench_window["Close"].iloc[0]) - 1
        n_years   = (bench_window.index[-1] - bench_window.index[0]).days / 365.25
        spy_ann   = (1 + spy_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    else:
        spy_total = spy_ann = 0

    win_rate   = (trades["net_return"] > 0).mean()
    avg_return = trades["net_return"].mean()
    avg_win    = trades.loc[trades["net_return"] > 0,  "net_return"].mean()
    avg_loss   = trades.loc[trades["net_return"] <= 0, "net_return"].mean()
    t_stat, p_value = stats.ttest_1samp(trades["net_return"], 0.0)

    tier_stats = {}
    if "cap_tier" in trades.columns:
        tier_stats = trades.groupby("cap_tier")["net_return"].agg(
            ["count", "mean"]
        ).round(4).to_dict()

    return {
        "total_trades":        len(trades),
        "win_rate":            round(float(win_rate), 4),
        "avg_return_per_trade": round(float(avg_return), 5),
        "avg_win":             round(float(avg_win), 5),
        "avg_loss":            round(float(avg_loss), 5),
        "avg_hold_days":       round(float(trades["hold_days"].mean()), 1),
        "exit_reasons":        trades["exit_reason"].value_counts().to_dict(),
        "trades_by_cap_tier":  tier_stats,
        "spy_annualized_return": round(float(spy_ann), 4),
        "t_stat":              round(float(t_stat), 3),
        "p_value":             round(float(p_value), 4),
        "significant_at_5pct": bool(p_value < 0.05),
    }


def simulate_portfolio(trades, benchmark_df,
                        starting_capital=100_000.0,
                        max_positions=15):
    if trades.empty:
        return {"error": "No trades to simulate."}

    bench = normalize_df(benchmark_df.copy())
    all_dates = sorted(set(list(trades["entry_date"]) + list(trades["exit_date"])))
    equity = starting_capital
    equity_curve = []
    position_size = starting_capital / max_positions

    for date in all_dates:
        closing = trades[trades["exit_date"] == date]
        for _, trade in closing.iterrows():
            equity += position_size * trade["net_return"]
        active_n = len(trades[
            (trades["entry_date"] <= date) & (trades["exit_date"] > date)
        ])
        equity_curve.append({"date": date, "equity": equity,
                              "n_positions": active_n})

    equity_df  = pd.DataFrame(equity_curve).set_index("date")
    total_ret  = (equity - starting_capital) / starting_capital
    start      = trades["entry_date"].min()
    end        = trades["exit_date"].max()
    bench_win  = bench[(bench.index >= start) & (bench.index <= end)]

    if len(bench_win) > 1:
        spy_total = (bench_win["Close"].iloc[-1] / bench_win["Close"].iloc[0]) - 1
        n_years   = (bench_win.index[-1] - bench_win.index[0]).days / 365.25
        spy_ann   = (1 + spy_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    else:
        spy_total = spy_ann = 0

    n_years      = (end - start).days / 365.25 if (end - start).days > 0 else 1
    strategy_ann = (1 + total_ret) ** (1 / n_years) - 1
    eq_series    = equity_df["equity"]
    max_dd       = ((eq_series - eq_series.cummax()) / eq_series.cummax()).min()

    return {
        "starting_capital":            starting_capital,
        "ending_capital":              round(equity, 2),
        "total_return":                round(total_ret, 4),
        "strategy_annualized_return":  round(strategy_ann, 4),
        "spy_annualized_return":       round(spy_ann, 4),
        "excess_return_vs_spy":        round(strategy_ann - spy_ann, 4),
        "beating_market_by_5pct":      bool((strategy_ann - spy_ann) >= 0.05),
        "max_drawdown":                round(float(max_dd), 4),
        "avg_positions_held":          round(equity_df["n_positions"].mean(), 1),
    }


def simulate_portfolio_vol_adjusted(trades: pd.DataFrame,
                                     benchmark_df: pd.DataFrame,
                                     position_sizes: pd.Series,
                                     starting_capital: float = 100_000.0) -> dict:
    """
    Portfolio simulation with volatility-adjusted position sizing.

    Instead of equal capital per position, each position is sized based
    on its own ATR -- lower volatility stocks get more capital, higher
    volatility stocks get less, so each contributes roughly equal RISK
    to the portfolio rather than equal CAPITAL.

    position_sizes: Series indexed by trade row index, containing the
    dollar amount allocated to each trade.
    """
    if trades.empty:
        return {"error": "No trades to simulate."}

    bench = normalize_df(benchmark_df.copy())
    trades_with_size = trades.copy()
    trades_with_size["position_size"] = position_sizes

    all_dates = sorted(set(
        list(trades["entry_date"]) + list(trades["exit_date"])
    ))

    equity       = starting_capital
    equity_curve = []

    for date in all_dates:
        closing = trades_with_size[trades_with_size["exit_date"] == date]
        for _, trade in closing.iterrows():
            pnl     = trade["position_size"] * trade["net_return"]
            equity += pnl

        active_n = len(trades[
            (trades["entry_date"] <= date) & (trades["exit_date"] > date)
        ])
        equity_curve.append({"date": date, "equity": equity,
                              "n_positions": active_n})

    equity_df  = pd.DataFrame(equity_curve).set_index("date")
    total_ret  = (equity - starting_capital) / starting_capital
    start      = trades["entry_date"].min()
    end        = trades["exit_date"].max()
    bench_win  = bench[(bench.index >= start) & (bench.index <= end)]

    if len(bench_win) > 1:
        spy_total = (bench_win["Close"].iloc[-1] / bench_win["Close"].iloc[0]) - 1
        n_years   = (bench_win.index[-1] - bench_win.index[0]).days / 365.25
        spy_ann   = (1 + spy_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    else:
        spy_total = spy_ann = 0

    n_years      = (end - start).days / 365.25 if (end - start).days > 0 else 1
    strategy_ann = (1 + total_ret) ** (1 / n_years) - 1
    eq_series    = equity_df["equity"]
    max_dd       = ((eq_series - eq_series.cummax()) / eq_series.cummax()).min()

    avg_pos_size = trades_with_size["position_size"].mean()
    min_pos_size = trades_with_size["position_size"].min()
    max_pos_size = trades_with_size["position_size"].max()

    return {
        "starting_capital":            starting_capital,
        "ending_capital":              round(equity, 2),
        "total_return":                round(total_ret, 4),
        "strategy_annualized_return":  round(strategy_ann, 4),
        "spy_annualized_return":       round(spy_ann, 4),
        "excess_return_vs_spy":        round(strategy_ann - spy_ann, 4),
        "beating_market_by_5pct":      bool((strategy_ann - spy_ann) >= 0.05),
        "max_drawdown":                round(float(max_dd), 4),
        "avg_positions_held":          round(equity_df["n_positions"].mean(), 1),
        "avg_position_size_dollars":   round(avg_pos_size, 0),
        "min_position_size_dollars":   round(min_pos_size, 0),
        "max_position_size_dollars":   round(max_pos_size, 0),
        "note": (
            "Volatility-adjusted sizing: each position sized so 1x ATR(14) "
            "move = 1% of starting capital. Low-vol stocks get more capital, "
            "high-vol stocks get less. Capped at 10% of portfolio per position."
        ),
    }


def simulate_portfolio_risk_sized(trades: pd.DataFrame,
                                   benchmark_df: pd.DataFrame,
                                   price_data: dict,
                                   spy_df: pd.DataFrame,
                                   starting_capital: float = 100_000.0,
                                   target_risk_pct: float = 0.01) -> dict:
    """
    Portfolio simulation with ATR + beta position sizing.
    No arbitrary caps -- risk model governs everything.
    """
    if trades.empty:
        return {"error": "No trades to simulate."}

    from signals.position_sizing import size_all_trades

    # Add sizing columns to trades
    sized_trades = size_all_trades(
        trades, price_data, spy_df,
        starting_capital   = starting_capital,
        target_risk_pct    = target_risk_pct,
    )

    bench      = normalize_df(benchmark_df.copy())
    all_dates  = sorted(set(
        list(sized_trades["entry_date"]) + list(sized_trades["exit_date"])
    ))

    equity       = starting_capital
    equity_curve = []

    for date in all_dates:
        closing = sized_trades[sized_trades["exit_date"] == date]
        for _, trade in closing.iterrows():
            pnl     = trade["position_size_dollars"] * trade["net_return"]
            equity += pnl
        active_n = len(sized_trades[
            (sized_trades["entry_date"] <= date) &
            (sized_trades["exit_date"]  >  date)
        ])
        equity_curve.append({"date": date, "equity": equity,
                              "n_positions": active_n})

    equity_df  = pd.DataFrame(equity_curve).set_index("date")
    total_ret  = (equity - starting_capital) / starting_capital
    start      = sized_trades["entry_date"].min()
    end        = sized_trades["exit_date"].max()
    bench_win  = bench[(bench.index >= start) & (bench.index <= end)]

    if len(bench_win) > 1:
        spy_total = (bench_win["Close"].iloc[-1] / bench_win["Close"].iloc[0]) - 1
        n_years   = (bench_win.index[-1] - bench_win.index[0]).days / 365.25
        spy_ann   = (1 + spy_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    else:
        spy_ann = 0

    n_years = (end - start).days / 365.25 if (end - start).days > 0 else 1
    # Guard: if total_ret < -1 the portfolio went negative (leverage blowup).
    # Report the raw total return rather than an undefined annualized figure.
    if total_ret <= -1:
        strategy_ann = float('nan')
    else:
        strategy_ann = (1 + total_ret) ** (1 / n_years) - 1
    eq_series    = equity_df["equity"]
    max_dd       = ((eq_series - eq_series.cummax()) / eq_series.cummax()).min()

    import math
    excess = (strategy_ann - spy_ann) if not math.isnan(strategy_ann) else float('nan')
    beating = bool(excess >= 0.05) if not math.isnan(excess) else False

    return {
        "starting_capital":            starting_capital,
        "ending_capital":              round(equity, 2),
        "total_return":                round(total_ret, 4),
        "strategy_annualized_return":  round(strategy_ann, 4) if not math.isnan(strategy_ann) else "n/a (portfolio went negative -- reduce target_risk_pct)",
        "spy_annualized_return":       round(spy_ann, 4),
        "excess_return_vs_spy":        round(excess, 4) if not math.isnan(excess) else "n/a",
        "beating_market_by_5pct":      beating,
        "max_drawdown":                round(float(max_dd), 4),
        "avg_positions_held":          round(equity_df["n_positions"].mean(), 1),
        "avg_beta":                    round(float(sized_trades["beta"].mean()), 3),
        "avg_position_size_dollars":   round(float(sized_trades["position_size_dollars"].mean()), 0),
        "min_position_size_dollars":   round(float(sized_trades["position_size_dollars"].min()), 0),
        "max_position_size_dollars":   round(float(sized_trades["position_size_dollars"].max()), 0),
        "note": (
            f"ATR+beta sizing: position = (portfolio × {target_risk_pct:.1%}) "
            "/ ATR_pct / beta. Max 20% per position (concentration limit). "
            "If portfolio went negative, reduce TARGET_RISK_PCT in run_backtest.py."
        ),
    }
