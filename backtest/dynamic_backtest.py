"""
Dynamic position management backtest.

Entry: same as indicator_backtest -- momentum percentile + MACD fresh
       cross + MA50 + volume + SAR gap + acceleration filter.

Position lifecycle (the new part):
  Each day, for every open position, we check:
    1. Should we ADD more? (pyramid) -- if price moved 1x ATR up since
       last add, SAR gap still wide, acceleration still positive
    2. Should we REDUCE? (scale out) -- if acceleration turned negative
       or SAR gap narrowed below 0.5x ATR
    3. Should we CLOSE fully? (exit) -- SAR exit or Chandelier exit

Capital released from scale-outs goes back into the portfolio pool and
can be redeployed into new entries or further pyramiding the same day.

Position sizing:
  Initial size: ATR+beta formula (same as before)
  Each pyramid add: 50% of the ORIGINAL position size
  Maximum 2 pyramids (so a position can grow to 2x original)
  Scale-out reduces current units by the computed fraction
"""

import pandas as pd
import numpy as np
from signals.classic_momentum import compute_12_1_momentum, compute_momentum_acceleration
from signals.entry_exit import (
    compute_entry_indicators, check_entry, get_momentum_threshold,
    compute_exit_indicators, compute_parabolic_sar, compute_chandelier_exit
)
from signals.regime import classify_regime
from signals.position_sizing import compute_position_size, compute_rolling_beta
from signals.dynamic_sizing import check_pyramid_conditions, check_scale_out_conditions
from signals.quality_filter import CoolingOffTracker, compute_quality_flags


def normalize_df(df):
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df


class DynamicBacktestConfig:
    def __init__(self,
                 momentum_pct_threshold: float = 0.20,
                 target_risk_pct: float = 0.005,
                 max_pyramids: int = 2,
                 cooldown_days: int = 60,
                 slippage_bps: float = 10.0,
                 commission_per_trade: float = 1.0):
        self.momentum_pct_threshold = momentum_pct_threshold
        self.target_risk_pct        = target_risk_pct
        self.max_pyramids           = max_pyramids
        self.cooldown_days          = cooldown_days
        self.slippage_bps           = slippage_bps
        self.commission_per_trade   = commission_per_trade


def run_dynamic_backtest(price_data: dict,
                          benchmark_df: pd.DataFrame,
                          spy_df: pd.DataFrame,
                          config: DynamicBacktestConfig,
                          market_caps: dict = None,
                          allowed_symbols: set = None,
                          starting_capital: float = 100_000.0,
                          fx_mode: bool = False,
                          tbill_annual_yield: float = 0.045) -> tuple:
    """
    Returns (trades_df, portfolio_df) where:
      trades_df:    one row per completed momentum trade (entry to full exit)
      portfolio_df: daily equity curve including idle capital returns

    Idle capital deployment:
      - Trending regime: idle cash invested in SPY (market return on idle slots)
      - Choppy regime:   idle cash earns T-bill yield (risk-free rate)
    This means the portfolio is always fully invested -- no dead cash.
    """
    price_data   = {t: normalize_df(df) for t, df in price_data.items()}
    benchmark_df = normalize_df(benchmark_df)
    spy_norm     = normalize_df(spy_df.copy())
    # Normalize SPY for idle capital deployment
    spy_norm = normalize_df(spy_df.copy())
    spy_returns = spy_norm["Close"].pct_change().fillna(0)

    # Daily T-bill return (compounded daily from annual yield)
    tbill_daily = (1 + tbill_annual_yield) ** (1/252) - 1

    market_caps  = market_caps or {}

    # Apply quality filter (short interest screening for equities)
    if allowed_symbols is not None:
        price_data = {t: df for t, df in price_data.items() if t in allowed_symbols}
        print(f"[dynamic-bt] {len(price_data)} symbols after quality filter")

    # Cooling-off tracker: prevents re-entry after stop-outs
    cooldown = CoolingOffTracker(cooldown_days=config.cooldown_days)

    regime_series = classify_regime(benchmark_df)
    if regime_series.index.tz is not None:
        regime_series.index = regime_series.index.tz_localize(None)
    regime_series.index = regime_series.index.normalize()

    print("[dynamic-bt] Pre-computing entry indicators...")
    entry_ind = {
        t: compute_entry_indicators(df, market_cap=market_caps.get(t))
        for t, df in price_data.items()
    }

    print("[dynamic-bt] Pre-computing quality flags (MA slope + 52w high)...")
    quality_flags = {t: compute_quality_flags(df) for t, df in price_data.items()}

    print("[dynamic-bt] Pre-computing exit indicators...")
    exit_ind = {t: pd.DataFrame({
        "close":      df["Close"],
        "sar":        compute_parabolic_sar(df),
        "chandelier": compute_chandelier_exit(df),
    }, index=df.index) for t, df in price_data.items()}

    print("[dynamic-bt] Pre-computing momentum and acceleration scores...")
    momentum_scores = {t: compute_12_1_momentum(df) for t, df in price_data.items()}
    accel_scores    = {t: compute_momentum_acceleration(df) for t, df in price_data.items()}

    print("[dynamic-bt] Pre-computing rolling betas...")
    betas = {}
    for t, df in price_data.items():
        try:
            betas[t] = compute_rolling_beta(df, spy_norm, window=63)
        except Exception:
            betas[t] = pd.Series(1.0, index=df.index)

    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    slip      = config.slippage_bps / 10_000.0

    # Portfolio state
    cash             = starting_capital
    open_positions   = {}
    # open_positions[ticker] = {
    #   "units": float,            current dollar exposure
    #   "original_size": float,    initial position size (for pyramid sizing)
    #   "entry_price": float,      first entry price
    #   "last_add_price": float,   price at most recent add
    #   "pyramid_count": int,      how many pyramids done
    #   "entry_date": date,
    #   "cap_tier": str,
    #   "total_cost": float,       total capital deployed (for P&L)
    # }

    completed_trades = []
    equity_curve     = []

    print(f"[dynamic-bt] Scanning {len(all_dates)} trading days...")
    for date in all_dates:
        daily_pnl = 0.0

        # ── 1. Check exits and scale-outs for all open positions ─────────
        to_close  = []
        to_adjust = []

        for ticker, pos in open_positions.items():
            df = price_data[ticker]
            if date not in df.index:
                continue

            close     = df.loc[date, "Close"]
            sar       = exit_ind[ticker].loc[date, "sar"]  if date in exit_ind[ticker].index else None
            chandelier = exit_ind[ticker].loc[date, "chandelier"] if date in exit_ind[ticker].index else None

            # Full exit: SAR or Chandelier
            if sar is not None and close < sar:
                to_close.append((ticker, close, "sar_exit"))
                continue
            if chandelier is not None and not pd.isna(chandelier) and close < chandelier:
                to_close.append((ticker, close, "chandelier_exit"))
                continue

            # Partial scale-out check
            accel_s = accel_scores.get(ticker, pd.Series(dtype=float))
            scale_frac, scale_reason = check_scale_out_conditions(
                entry_ind[ticker], accel_s, date
            )
            if scale_frac > 0 and pos["units"] > 0:
                to_adjust.append((ticker, scale_frac, scale_reason, close))

        # Process full exits
        for ticker, exit_price, reason in to_close:
            pos     = open_positions.pop(ticker)
            xp      = exit_price * (1 - slip)
            ret     = (xp - pos["entry_price"]) / pos["entry_price"]
            pnl     = pos["units"] * ret - config.commission_per_trade / 1000.0 * pos["units"]
            cash   += pos["units"] + pnl
            daily_pnl += pnl
            # Register cooling-off for stop-outs (not for end_of_backtest)
            if reason in ("sar_exit", "chandelier_exit"):
                cooldown.register_stopout(ticker, date)
            completed_trades.append({
                "ticker":      ticker,
                "cap_tier":    pos["cap_tier"],
                "entry_date":  pos["entry_date"],
                "exit_date":   date,
                "entry_price": pos["entry_price"],
                "exit_price":  xp,
                "exit_reason": reason,
                "pyramids_done": pos["pyramid_count"],
                "net_return":  ret,
                "hold_days":   (date - pos["entry_date"]).days,
            })

        # Process partial scale-outs
        for ticker, frac, reason, close in to_adjust:
            pos            = open_positions[ticker]
            reduce_units   = pos["units"] * frac
            xp             = close * (1 - slip)
            ret            = (xp - pos["entry_price"]) / pos["entry_price"]
            pnl            = reduce_units * ret
            cash          += reduce_units + pnl
            daily_pnl     += pnl
            pos["units"]  -= reduce_units

        # ── 2. Regime check ───────────────────────────────────────────────
        if date not in regime_series.index or regime_series.loc[date] != "trending":
            equity_val = cash + sum(
                p["units"] * (price_data[t].loc[date, "Close"] if date in price_data[t].index else 0)
                for t, p in open_positions.items()
            )
            equity_curve.append({"date": date, "equity": equity_val,
                                  "cash": cash, "n_positions": len(open_positions),
                                  "idle_deployed": "tbill"})
            continue

        # ── 3. Pyramid existing winners ───────────────────────────────────
        for ticker, pos in list(open_positions.items()):
            if pos["pyramid_count"] >= config.max_pyramids:
                continue
            df = price_data[ticker]
            if date not in df.index:
                continue

            accel_s = accel_scores.get(ticker, pd.Series(dtype=float))
            if check_pyramid_conditions(
                df, exit_ind[ticker], entry_ind[ticker], accel_s,
                date, pos["last_add_price"]
            ):
                add_size = pos["original_size"] * 0.5
                if add_size <= cash:
                    add_price        = df.loc[date, "Close"] * (1 + slip)
                    cash            -= add_size
                    pos["units"]    += add_size
                    pos["last_add_price"] = add_price
                    pos["pyramid_count"] += 1

        # ── 4. New entries ────────────────────────────────────────────────
        mom_scores_today = {}
        for t, scores in momentum_scores.items():
            if t in open_positions:
                continue
            if date not in scores.index:
                continue
            val = scores.loc[date]
            if not pd.isna(val):
                mom_scores_today[t] = val

        if mom_scores_today:
            all_vals    = list(mom_scores_today.values())
            candidates  = sorted(mom_scores_today.items(), key=lambda x: x[1], reverse=True)

            for ticker, mom_score in candidates:
                if cash < starting_capital * 0.02:  # less than 2% cash left
                    break

                # Tier-specific momentum threshold
                eff_thresh    = get_momentum_threshold(
                    entry_ind[ticker], date, config.momentum_pct_threshold
                )
                thresh_val    = np.percentile(all_vals, (1 - eff_thresh) * 100)
                if mom_score < thresh_val:
                    continue

                # Cooling-off: skip if recently stopped out of this symbol
                if not cooldown.is_allowed(ticker, date):
                    continue

                # Quality filter: 200-day MA slope + 52-week high proximity
                # For FX, skip the 52w high filter (pairs oscillate around
                # equilibrium and don't exhibit the same structural uptrend
                # behaviour as equities or crypto)
                if ticker in quality_flags and date in quality_flags[ticker].index:
                    qrow = quality_flags[ticker].loc[date]
                    if fx_mode:
                        if not bool(qrow.get("ma200_slope_ok", False)):
                            continue
                    else:
                        if not bool(qrow.get("quality_ok", False)):
                            continue

                # Acceleration filter
                accel_s = accel_scores.get(ticker, pd.Series(dtype=float))
                if date in accel_s.index:
                    accel = accel_s.loc[date]
                    if pd.isna(accel) or accel <= 0:
                        continue

                # Indicator entry filter
                if not check_entry(entry_ind[ticker], date):
                    continue

                df = price_data[ticker]
                if date not in df.index:
                    continue

                entry_price = df.loc[date, "Close"] * (1 + slip)

                # ATR + beta position size
                atr14 = entry_ind[ticker].loc[date, "atr14"] if date in entry_ind[ticker].index else None
                beta  = betas[ticker].loc[date] if ticker in betas and date in betas[ticker].index else 1.0
                if atr14 is None or pd.isna(atr14):
                    continue

                size = compute_position_size(
                    portfolio_value = cash + sum(
                        p["units"] for p in open_positions.values()
                    ),
                    target_risk_pct = config.target_risk_pct,
                    atr14           = atr14,
                    entry_price     = entry_price,
                    beta            = float(beta) if not pd.isna(beta) else 1.0,
                    max_position_pct = 0.20,
                )
                size = min(size, cash)
                if size < 100:  # minimum $100 position
                    continue

                cash -= size
                cap_tier = entry_ind[ticker].loc[date, "cap_tier"] if date in entry_ind[ticker].index else "unknown"
                open_positions[ticker] = {
                    "units":          size,
                    "original_size":  size,
                    "entry_price":    entry_price,
                    "last_add_price": entry_price,
                    "pyramid_count":  0,
                    "entry_date":     date,
                    "cap_tier":       cap_tier,
                }

        # ── 5. Idle capital return (SPY if trending, T-bill if choppy) ──────
        is_trending = (date in regime_series.index and
                       regime_series.loc[date] == "trending")

        # Deployed capital = cash already used for open positions
        deployed = sum(p["units"] for p in open_positions.values())
        idle_cash = max(cash, 0)  # uninvested cash

        if idle_cash > 0:
            if is_trending and date in spy_returns.index:
                # Idle cash tracks SPY return
                spy_ret = spy_returns.loc[date]
                idle_pnl = idle_cash * spy_ret
            else:
                # Idle cash earns T-bill rate
                idle_pnl = idle_cash * tbill_daily
            cash += idle_pnl
        else:
            idle_pnl = 0.0

        # ── 6. Mark-to-market equity ──────────────────────────────────────
        mtm = sum(
            p["units"] * (price_data[t].loc[date, "Close"] / price_data[t].loc[
                p["entry_date"], "Close"] if date in price_data[t].index and p["entry_date"] in price_data[t].index else 1.0)
            for t, p in open_positions.items()
        )
        equity_val = cash + mtm
        equity_curve.append({
            "date":          date,
            "equity":        equity_val,
            "cash":          cash,
            "n_positions":   len(open_positions),
            "idle_deployed": "spy" if is_trending else "tbill",
        })

    # Close remaining positions at end of data
    last_date = all_dates[-1]
    for ticker, pos in open_positions.items():
        df = price_data[ticker]
        if last_date in df.index:
            xp   = df.loc[last_date, "Close"] * (1 - slip)
            ret  = (xp - pos["entry_price"]) / pos["entry_price"]
            pnl  = pos["units"] * ret
            cash += pos["units"] + pnl
            completed_trades.append({
                "ticker":        ticker,
                "cap_tier":      pos["cap_tier"],
                "entry_date":    pos["entry_date"],
                "exit_date":     last_date,
                "entry_price":   pos["entry_price"],
                "exit_price":    xp,
                "exit_reason":   "end_of_backtest",
                "pyramids_done": pos["pyramid_count"],
                "net_return":    ret,
                "hold_days":     (last_date - pos["entry_date"]).days,
            })

    trades_df    = pd.DataFrame(completed_trades)
    portfolio_df = pd.DataFrame(equity_curve).set_index("date")
    return trades_df, portfolio_df


def summarize_dynamic_results(trades_df: pd.DataFrame,
                               portfolio_df: pd.DataFrame,
                               benchmark_df: pd.DataFrame,
                               starting_capital: float = 100_000.0) -> dict:
    if trades_df.empty:
        return {"error": "No trades generated."}

    bench = normalize_df(benchmark_df.copy())
    start = portfolio_df.index.min()
    end   = portfolio_df.index.max()
    bench_win = bench[(bench.index >= start) & (bench.index <= end)]

    if len(bench_win) > 1:
        spy_total = (bench_win["Close"].iloc[-1] / bench_win["Close"].iloc[0]) - 1
        n_years   = (bench_win.index[-1] - bench_win.index[0]).days / 365.25
        spy_ann   = (1 + spy_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    else:
        spy_ann = 0

    final_equity = portfolio_df["equity"].iloc[-1]
    total_ret    = (final_equity - starting_capital) / starting_capital
    n_years      = (end - start).days / 365.25 if (end - start).days > 0 else 1

    if total_ret <= -1:
        strategy_ann = float("nan")
    else:
        strategy_ann = (1 + total_ret) ** (1 / n_years) - 1

    eq           = portfolio_df["equity"]
    max_dd       = ((eq - eq.cummax()) / eq.cummax()).min()

    import math
    from scipy import stats
    win_rate     = (trades_df["net_return"] > 0).mean()
    avg_ret      = trades_df["net_return"].mean()
    avg_win      = trades_df.loc[trades_df["net_return"] > 0, "net_return"].mean()
    avg_loss     = trades_df.loc[trades_df["net_return"] <= 0, "net_return"].mean()
    t_stat, p_val = stats.ttest_1samp(trades_df["net_return"], 0.0)
    pyramided    = (trades_df["pyramids_done"] > 0).sum()

    excess = (strategy_ann - spy_ann) if not math.isnan(strategy_ann) else float("nan")

    return {
        "total_trades":                   len(trades_df),
        "trades_with_pyramids":           int(pyramided),
        "win_rate":                       round(float(win_rate), 4),
        "avg_return_per_trade":           round(float(avg_ret), 5),
        "avg_win":                        round(float(avg_win), 5),
        "avg_loss":                       round(float(avg_loss), 5),
        "avg_hold_days":                  round(float(trades_df["hold_days"].mean()), 1),
        "exit_reasons":                   trades_df["exit_reason"].value_counts().to_dict(),
        "starting_capital":               starting_capital,
        "ending_capital":                 round(float(final_equity), 2),
        "total_return":                   round(float(total_ret), 4),
        "strategy_annualized_return":     round(float(strategy_ann), 4) if not math.isnan(strategy_ann) else "n/a",
        "spy_annualized_return":          round(float(spy_ann), 4),
        "excess_return_vs_spy":           round(float(excess), 4) if not math.isnan(excess) else "n/a",
        "beating_market_by_5pct":         bool(excess >= 0.05) if not math.isnan(excess) else False,
        "max_drawdown":                   round(float(max_dd), 4),
        "avg_positions_held":             round(float(portfolio_df["n_positions"].mean()), 1),
        "avg_cash_pct":                   round(float((portfolio_df["cash"] / portfolio_df["equity"]).mean()), 3),
        "idle_in_spy_pct":                round(float((portfolio_df.get("idle_deployed", pd.Series()) == "spy").mean()), 3) if "idle_deployed" in portfolio_df.columns else "n/a",
        "t_stat":                         round(float(t_stat), 3),
        "p_value":                        round(float(p_val), 4),
        "significant_at_5pct":            bool(p_val < 0.05),
        "note":                           "Idle capital: SPY during trending regime, T-bills (4.5%) during choppy regime.",
    }
