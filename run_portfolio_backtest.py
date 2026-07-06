"""
Portfolio Backtester
====================

Simulates the live order-generation logic over history:
  - weekly / 5-trading-day rescreening
  - probability + signal filters
  - SPY 200-day regime filter
  - ATR + beta position sizing
  - 15% max position cap
  - 100% max portfolio deployment
  - IBKR-style estimated commissions: $0.005/share, $1.00 minimum
  - 2% minimum final position size
  - optional stop exit using the same displayed stop reference

Outputs:
  - portfolio_backtest_equity.csv
  - portfolio_backtest_trades.csv
  - portfolio_backtest_summary.csv

This is a research tool, not investment advice.
"""

import os
import sys
import math
import random
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.provider import YFinanceProvider
from data.universe import get_universe, get_market_caps
from data.screener import prefilter_tickers
from screener_5pct import (
    MIN_WEEKLY_VOL,
    MAX_WEEKLY_VOL,
    MIN_MARKET_CAP,
    MAX_MARKET_CAP,
    MIN_AVG_DOLLAR_VOL,
    compute_weekly_vol,
    compute_3m_momentum,
    detect_signals,
    compute_signal_boost,
    compute_prob_5pct,
    SIGNAL_ADJUSTMENTS,
)
from generate_orders import (
    TARGET_RISK_PCT,
    MAX_POSITION_PCT,
    MAX_DEPLOY_PCT,
    MIN_POSITION_PCT,
    COMMISSION_PER_SHARE,
    MIN_COMMISSION,
    MIN_PROB,
    MIN_SIGNALS,
    MIN_SAR_GAP,
    flatten_df,
    normalize_df,
    compute_atr14,
    compute_stop_levels,
)
from signals.position_sizing import compute_rolling_beta


# ── Backtest configuration ────────────────────────────────────────────────────

try:
    from config import (
        INITIAL_CAPITAL,
        BACKTEST_YEARS,
        RESCREEN_DAYS,
        HOLDING_DAYS,
        MAX_TICKERS,
        USE_STOPS,
        BENCHMARK,
        RANDOM_SEED,
    )
except Exception:
    INITIAL_CAPITAL = 100_000.0
    BACKTEST_YEARS = 3
    RESCREEN_DAYS = 5
    HOLDING_DAYS = 5
    MAX_TICKERS = 900
    USE_STOPS = True
    BENCHMARK = "SPY"
    RANDOM_SEED = 42


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    summary: pd.DataFrame


def estimate_commission(shares: int) -> float:
    if shares <= 0:
        return 0.0
    return round(max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE), 2)


def position_size(equity: float, price: float, atr14: float, beta: float) -> float:
    if price <= 0 or atr14 <= 0:
        return equity * 0.05
    atr_pct = atr14 / price
    beta_eff = max(abs(beta), 0.3)
    raw = (equity * TARGET_RISK_PCT) / atr_pct / beta_eff
    return round(min(raw, equity * MAX_POSITION_PCT), 2)


def max_shares_affordable(remaining_cash: float, price: float, requested_shares: int) -> int:
    shares = max(0, min(int(requested_shares), int(remaining_cash // price)))
    while shares > 0:
        cost = shares * price + estimate_commission(shares)
        if cost <= remaining_cash:
            return shares
        shares -= 1
    return 0


def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def summarize_performance(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return pd.DataFrame([{"error": "No equity curve generated"}])

    equity = equity_df.set_index("date")["equity"].astype(float)
    daily = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    max_dd = compute_drawdown(equity).min()
    sharpe = (daily.mean() / daily.std() * math.sqrt(252)) if daily.std() > 0 else np.nan
    downside = daily[daily < 0]
    sortino = (daily.mean() / downside.std() * math.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else np.nan

    if trades_df.empty:
        win_rate = avg_trade = avg_win = avg_loss = profit_factor = np.nan
        n_trades = 0
    else:
        rets = trades_df["net_return_pct"].astype(float) / 100.0
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        win_rate = len(wins) / len(rets) if len(rets) else np.nan
        avg_trade = rets.mean() if len(rets) else np.nan
        avg_win = wins.mean() if len(wins) else np.nan
        avg_loss = losses.mean() if len(losses) else np.nan
        gross_profit = trades_df.loc[trades_df["pnl_$"] > 0, "pnl_$"].sum()
        gross_loss = -trades_df.loc[trades_df["pnl_$"] <= 0, "pnl_$"].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
        n_trades = len(trades_df)

    row = {
        "start_date": str(equity.index[0].date()),
        "end_date": str(equity.index[-1].date()),
        "initial_capital": round(float(equity.iloc[0]), 2),
        "ending_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(float(total_return * 100), 2),
        "cagr_pct": round(float(cagr * 100), 2),
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "sharpe": round(float(sharpe), 2) if not pd.isna(sharpe) else np.nan,
        "sortino": round(float(sortino), 2) if not pd.isna(sortino) else np.nan,
        "trades": int(n_trades),
        "win_rate_pct": round(float(win_rate * 100), 2) if not pd.isna(win_rate) else np.nan,
        "avg_trade_pct": round(float(avg_trade * 100), 2) if not pd.isna(avg_trade) else np.nan,
        "avg_win_pct": round(float(avg_win * 100), 2) if not pd.isna(avg_win) else np.nan,
        "avg_loss_pct": round(float(avg_loss * 100), 2) if not pd.isna(avg_loss) else np.nan,
        "profit_factor": round(float(profit_factor), 2) if not pd.isna(profit_factor) else np.nan,
    }
    return pd.DataFrame([row])


def prepare_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, float]]:
    provider = YFinanceProvider()
    tickers = get_universe(include_midcap=True, mode="broad")
    if len(tickers) > MAX_TICKERS:
        random.seed(RANDOM_SEED)
        tickers = random.sample(tickers, MAX_TICKERS)

    print(f"[1/4] Pre-filtering {len(tickers)} tickers...")
    tickers = prefilter_tickers(tickers)
    print(f"      {len(tickers)} common stocks after string filter")

    end = dt.date.today()
    start = end - dt.timedelta(days=int((BACKTEST_YEARS + 1.2) * 365))

    all_tickers = sorted(set(tickers + [BENCHMARK]))
    print(f"[2/4] Downloading daily history for {len(all_tickers)} tickers...")
    raw = provider.get_history_bulk(all_tickers, start=str(start), end=str(end), interval="1d")

    price_data = {}
    for t, df in raw.items():
        df = normalize_df(flatten_df(df))
        if df is None or df.empty or len(df) < 252:
            continue
        price_data[t] = df

    if BENCHMARK not in price_data:
        raise RuntimeError("Could not download SPY benchmark data")

    spy = price_data.pop(BENCHMARK)

    print(f"[3/4] Fetching market caps...")
    market_caps = get_market_caps(list(price_data.keys()))

    filtered = {}
    for t, df in price_data.items():
        if not (MIN_MARKET_CAP <= market_caps.get(t, 0) <= MAX_MARKET_CAP):
            continue
        recent = df.tail(20)
        if len(recent) < 20:
            continue
        avg_dollar_vol = (recent["Close"] * recent["Volume"]).mean()
        if avg_dollar_vol < MIN_AVG_DOLLAR_VOL:
            continue
        filtered[t] = df

    print(f"[4/4] {len(filtered)} stocks available for walk-forward simulation")
    return filtered, spy, market_caps


def rank_candidates(asof: pd.Timestamp, price_data: dict[str, pd.DataFrame], market_caps: dict[str, float]) -> pd.DataFrame:
    hist = {
        t: df[df.index <= asof]
        for t, df in price_data.items()
        if len(df[df.index <= asof]) >= 252
    }
    if len(hist) < 10:
        return pd.DataFrame()

    mom = {}
    for t, df in hist.items():
        try:
            score = compute_3m_momentum(df)
            if score is not None and not pd.isna(score):
                mom[t] = float(score)
        except Exception:
            pass
    if not mom:
        return pd.DataFrame()

    threshold = np.percentile(list(mom.values()), 80)
    top20 = {t for t, score in mom.items() if score >= threshold}

    rows = []
    for ticker, df in hist.items():
        try:
            wv = compute_weekly_vol(df)
            if not (MIN_WEEKLY_VOL <= wv <= MAX_WEEKLY_VOL):
                continue
            signals = detect_signals(df, ticker, top20)
            boost = compute_signal_boost(signals)
            prob = compute_prob_5pct(df, boost)
            if not prob:
                continue
            active = sum(1 for k in SIGNAL_ADJUSTMENTS if signals.get(k))
            rows.append({
                "ticker": ticker,
                "price": float(df["Close"].iloc[-1]),
                "weekly_vol_pct": prob["weekly_vol_pct"],
                "prob_up_5pct": float(prob["prob_up_5pct"]),
                "active_signals": int(active),
                "sar_gap_pct": float(signals.get("sar_gap_pct", 0.0)),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out[
        (out["prob_up_5pct"] >= MIN_PROB) &
        (out["active_signals"] >= MIN_SIGNALS) &
        (out["sar_gap_pct"] >= MIN_SAR_GAP)
    ].copy()
    if out.empty:
        return out
    return out.sort_values(["prob_up_5pct", "active_signals"], ascending=False).reset_index(drop=True)


def is_trending(spy_hist: pd.DataFrame) -> bool:
    if len(spy_hist) < 200:
        return False
    return bool(spy_hist["Close"].iloc[-1] > spy_hist["Close"].rolling(200).mean().iloc[-1])


def simulate_portfolio(price_data: dict[str, pd.DataFrame], spy: pd.DataFrame, market_caps: dict[str, float]) -> BacktestResult:
    all_dates = sorted(spy.index)
    start_cutoff = all_dates[-1] - pd.DateOffset(years=BACKTEST_YEARS)
    screen_dates = [d for d in all_dates if d >= start_cutoff][::RESCREEN_DAYS]

    equity = INITIAL_CAPITAL
    equity_records = []
    trade_records = []

    print(f"Simulating {len(screen_dates)} rebalance periods from {screen_dates[0].date()} to {screen_dates[-1].date()}...")

    for i, asof in enumerate(screen_dates[:-1], start=1):
        exit_date = screen_dates[i]
        spy_hist = spy[spy.index <= asof]

        if not is_trending(spy_hist):
            equity_records.append({"date": exit_date, "equity": round(equity, 2), "positions": 0, "period_return_pct": 0.0, "note": "regime_off"})
            continue

        candidates = rank_candidates(asof, price_data, market_caps)
        if candidates.empty:
            equity_records.append({"date": exit_date, "equity": round(equity, 2), "positions": 0, "period_return_pct": 0.0, "note": "no_candidates"})
            continue

        orders = []
        max_total_cost = equity * MAX_DEPLOY_PCT
        min_total_cost = equity * MIN_POSITION_PCT
        remaining_cash = max_total_cost

        spy_for_beta = spy[spy.index <= asof]
        for _, row in candidates.iterrows():
            ticker = row["ticker"]
            hist = price_data[ticker][price_data[ticker].index <= asof]
            if hist.empty:
                continue
            price = float(hist["Close"].iloc[-1])
            try:
                beta_s = compute_rolling_beta(hist, spy_for_beta, window=63)
                beta = float(beta_s.iloc[-1]) if len(beta_s) and not pd.isna(beta_s.iloc[-1]) else 1.0
            except Exception:
                beta = 1.0
            try:
                atr = compute_atr14(hist)
                sar_stop, chandelier_stop = compute_stop_levels(hist)
                stop_level = max(float(sar_stop), float(chandelier_stop))
            except Exception:
                atr = price * 0.05
                stop_level = price * 0.93

            target_value = position_size(equity, price, atr, beta)
            requested_shares = int(target_value / price)
            shares = max_shares_affordable(remaining_cash, price, requested_shares)
            if shares <= 0:
                continue
            buy_commission = estimate_commission(shares)
            buy_cost = round(shares * price + buy_commission, 2)
            if buy_cost < min_total_cost:
                continue

            orders.append({
                "ticker": ticker,
                "entry_date": asof,
                "entry_price": price,
                "shares": shares,
                "buy_commission_$": buy_commission,
                "buy_cost_$": buy_cost,
                "stop_level": stop_level,
                "prob_up_5pct": row["prob_up_5pct"],
                "active_signals": row["active_signals"],
            })
            remaining_cash -= buy_cost
            if remaining_cash < min_total_cost:
                break

        if not orders:
            equity_records.append({"date": exit_date, "equity": round(equity, 2), "positions": 0, "period_return_pct": 0.0, "note": "no_orders_after_cap"})
            continue

        start_equity = equity
        period_pnl = 0.0
        for o in orders:
            ticker = o["ticker"]
            df = price_data[ticker]
            future = df[(df.index > asof) & (df.index <= exit_date)]
            if future.empty:
                continue

            exit_reason = "time"
            exit_px = float(future["Close"].iloc[-1])
            actual_exit_date = future.index[-1]

            if USE_STOPS:
                breached = future[future["Low"] <= o["stop_level"]]
                if not breached.empty:
                    actual_exit_date = breached.index[0]
                    exit_px = float(o["stop_level"])
                    exit_reason = "stop"

            sell_commission = estimate_commission(int(o["shares"]))
            proceeds = o["shares"] * exit_px - sell_commission
            pnl = proceeds - o["buy_cost_$"]
            period_pnl += pnl

            trade_value = o["shares"] * o["entry_price"]
            net_ret = pnl / o["buy_cost_$"] if o["buy_cost_$"] else 0.0
            trade_records.append({
                "entry_date": str(pd.Timestamp(o["entry_date"]).date()),
                "exit_date": str(pd.Timestamp(actual_exit_date).date()),
                "ticker": ticker,
                "shares": int(o["shares"]),
                "entry_price": round(o["entry_price"], 2),
                "exit_price": round(exit_px, 2),
                "trade_value_$": round(trade_value, 2),
                "buy_commission_$": round(o["buy_commission_$"], 2),
                "sell_commission_$": round(sell_commission, 2),
                "pnl_$": round(pnl, 2),
                "net_return_pct": round(net_ret * 100, 2),
                "exit_reason": exit_reason,
                "prob_up_5pct": round(float(o["prob_up_5pct"]), 4),
                "active_signals": int(o["active_signals"]),
            })

        equity += period_pnl
        period_ret = (equity / start_equity - 1) if start_equity else 0.0
        equity_records.append({
            "date": exit_date,
            "equity": round(equity, 2),
            "positions": len(orders),
            "period_return_pct": round(period_ret * 100, 2),
            "note": "traded",
        })

        if i % 25 == 0:
            print(f"  {i}/{len(screen_dates)-1} periods | equity ${equity:,.0f}")

    equity_df = pd.DataFrame(equity_records)
    if not equity_df.empty:
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.sort_values("date")
    trades_df = pd.DataFrame(trade_records)
    summary_df = summarize_performance(equity_df, trades_df)
    return BacktestResult(equity_df, trades_df, summary_df)


def main():
    print("=" * 70)
    print("PORTFOLIO BACKTEST")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Lookback: {BACKTEST_YEARS} years | rebalance every {RESCREEN_DAYS} trading days")
    print(f"Commission: ${COMMISSION_PER_SHARE:.3f}/share, ${MIN_COMMISSION:.2f} minimum")
    print(f"Min position: {MIN_POSITION_PCT*100:.1f}% | Max position: {MAX_POSITION_PCT*100:.1f}%")
    print("=" * 70)

    price_data, spy, market_caps = prepare_data()
    if len(price_data) < 10:
        print("Not enough data to run a meaningful portfolio backtest.")
        return

    result = simulate_portfolio(price_data, spy, market_caps)

    result.equity_curve.to_csv("portfolio_backtest_equity.csv", index=False)
    result.trades.to_csv("portfolio_backtest_trades.csv", index=False)
    result.summary.to_csv("portfolio_backtest_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(result.summary.to_string(index=False))
    print("\nSaved:")
    print("  portfolio_backtest_equity.csv")
    print("  portfolio_backtest_trades.csv")
    print("  portfolio_backtest_summary.csv")
    print("\nNext: compare the CAGR, max drawdown, Sharpe, win rate, and profit factor to SPY.")


if __name__ == "__main__":
    main()
