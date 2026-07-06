"""
Order Generator
===============
Takes screener output and generates actual entry orders with position sizes.

Workflow:
  1. Load today's screener results (screener_results.csv)
  2. Filter to high-conviction candidates:
       - P(up 5% in 1 week) >= MIN_PROB
       - At least MIN_SIGNALS active signals
  3. For each candidate, compute position size using ATR + beta
  4. Check regime (SPY above 200-day MA) -- only enter in trending market
  5. Output a clean order table ready to execute manually or via IBKR API

Position sizing:
  Same ATR + beta formula as the rest of the system:
  position_size = (portfolio * TARGET_RISK_PCT) / ATR_pct / beta
  Capped at MAX_POSITION_PCT of portfolio per position, then capped again at the portfolio level including estimated commissions and a minimum position threshold.

Exit plan for each order:
  - Parabolic SAR (pre-computed, shown as a reference stop level)
  - Chandelier Exit = highest_close - 2x ATR (shown as reference)
  - Neither is a hard stop in this script -- they fire in live monitoring

Live trading note:
  When connected to IBKR via ib_async, replace the print statements
  at the bottom with actual order placement calls.
"""

import sys, os, datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.provider import YFinanceProvider
from signals.entry_exit import compute_parabolic_sar, compute_chandelier_exit
from signals.regime import classify_regime
from signals.position_sizing import compute_rolling_beta


# ── Configuration ──────────────────────────────────────────────────────────────

try:
    from config import (
        PORTFOLIO_VALUE,
        TARGET_RISK_PCT,
        MAX_POSITION_PCT,
        MAX_DEPLOY_PCT,
        MIN_POSITION_PCT,
        COMMISSION_PER_SHARE,
        MIN_COMMISSION,
        MIN_PROB,
        MIN_SIGNALS,
        MIN_SAR_GAP,
    )
except Exception:
    PORTFOLIO_VALUE  = 100_000.0
    TARGET_RISK_PCT  = 0.005
    MAX_POSITION_PCT = 0.15
    MAX_DEPLOY_PCT   = 1.00
    MIN_POSITION_PCT = 0.02
    COMMISSION_PER_SHARE = 0.005
    MIN_COMMISSION = 1.00
    MIN_PROB       = 0.30
    MIN_SIGNALS    = 2
    MIN_SAR_GAP    = 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_df(df):
    if df is None or df.empty:
        return df
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df


def flatten_df(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def check_regime(spy_df: pd.DataFrame) -> tuple[bool, str]:
    """Returns (is_trending, description)."""
    try:
        regime = classify_regime(spy_df)
        latest = regime.iloc[-1]
        return latest == "trending", latest
    except Exception:
        ma200 = spy_df["Close"].rolling(200).mean()
        is_above = spy_df["Close"].iloc[-1] > ma200.iloc[-1]
        return is_above, "trending" if is_above else "choppy"


def compute_position_size(price: float, atr14: float,
                           beta: float) -> float:
    """ATR + beta position sizing."""
    if atr14 <= 0 or price <= 0:
        return PORTFOLIO_VALUE * 0.05

    atr_pct        = atr14 / price
    beta_eff       = max(abs(beta), 0.3)
    size           = (PORTFOLIO_VALUE * TARGET_RISK_PCT) / atr_pct / beta_eff
    max_size       = PORTFOLIO_VALUE * MAX_POSITION_PCT
    return round(min(size, max_size), 2)


def estimate_commission(shares: int) -> float:
    """Estimate per-order commission: $0.005/share, $1.00 minimum."""
    if shares <= 0:
        return 0.0
    return round(max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE), 2)


def cap_orders_to_budget(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce a hard portfolio-level deployment cap including estimated commissions.

    Individual position sizing happens first using the ATR + beta formula.
    This second pass ranks orders by conviction and keeps or trims positions so
    total cost never exceeds PORTFOLIO_VALUE * MAX_DEPLOY_PCT.

    Commission model:
      - $0.005 per share
      - $1.00 minimum per order

    Minimum position rule:
      - Skip any final order whose total cost is below MIN_POSITION_PCT of portfolio.
      - This avoids tiny leftover trades such as buying only 1 share.
    """
    if orders_df is None or orders_df.empty:
        return orders_df

    max_total_cost = PORTFOLIO_VALUE * MAX_DEPLOY_PCT
    min_total_cost = PORTFOLIO_VALUE * MIN_POSITION_PCT

    sort_cols = [c for c in ["prob_up_5pct", "active_signals"] if c in orders_df.columns]
    if sort_cols:
        orders_df = orders_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    selected = []
    running_total_cost = 0.0

    for _, row in orders_df.iterrows():
        row = row.copy()
        remaining = max_total_cost - running_total_cost

        if remaining < min_total_cost:
            break

        price = float(row["price"])
        if price <= 0:
            continue

        requested_shares = int(row["shares"])
        shares = min(requested_shares, int(remaining // price))

        while shares > 0:
            trade_value = round(shares * price, 2)
            commission = estimate_commission(shares)
            total_cost = round(trade_value + commission, 2)

            if total_cost <= remaining:
                break
            shares -= 1

        if shares < 1:
            continue

        trade_value = round(shares * price, 2)
        commission = estimate_commission(shares)
        total_cost = round(trade_value + commission, 2)

        # Avoid tiny residual trades after trimming to fit the remaining budget.
        if total_cost < min_total_cost:
            continue

        row["shares"] = shares
        row["position_$"] = trade_value
        row["commission_$"] = commission
        row["total_cost_$"] = total_cost
        row["pct_portfolio"] = round(total_cost / PORTFOLIO_VALUE * 100, 1)

        selected.append(row)
        running_total_cost += total_cost

    if not selected:
        return orders_df.iloc[0:0].copy()

    return pd.DataFrame(selected).reset_index(drop=True)

def compute_atr14(df: pd.DataFrame) -> float:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=14, adjust=False).mean().iloc[-1])


def compute_stop_levels(df: pd.DataFrame) -> tuple[float, float]:
    """Returns (sar_stop, chandelier_stop)."""
    try:
        sar        = compute_parabolic_sar(df)
        chandelier = compute_chandelier_exit(df)
        return float(sar.iloc[-1]), float(chandelier.iloc[-1])
    except Exception:
        price = df["Close"].iloc[-1]
        atr   = compute_atr14(df)
        return price * 0.93, price - 2 * atr


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ORDER GENERATOR")
    print(f"Portfolio: ${PORTFOLIO_VALUE:,.0f}")
    print(f"Filters: P(up 5%) >= {MIN_PROB*100:.0f}%, "
          f"signals >= {MIN_SIGNALS}, SAR gap >= {MIN_SAR_GAP}%")
    print("=" * 60)

    # ── Load screener results ─────────────────────────────────────────────
    if not os.path.exists("screener_results.csv"):
        print("\nERROR: screener_results.csv not found.")
        print("Run screener_5pct.py first to generate candidates.")
        return

    screener = pd.read_csv("screener_results.csv")
    print(f"\nLoaded {len(screener)} screener results")

    # Apply filters
    candidates = screener[
        (screener["prob_up_5pct"] >= MIN_PROB) &
        (screener["active_signals"] >= MIN_SIGNALS) &
        (screener["sar_gap_pct"] >= MIN_SAR_GAP)
    ].copy()

    print(f"{len(candidates)} candidates after filters")

    if candidates.empty:
        print("\nNo candidates meet the criteria.")
        print(f"Try lowering MIN_PROB ({MIN_PROB}) or MIN_SIGNALS ({MIN_SIGNALS})")
        return

    # ── Check market regime ───────────────────────────────────────────────
    print("\nChecking market regime (SPY 200-day MA)...")
    provider = YFinanceProvider()
    spy_df   = provider.get_history("SPY", period="1y", interval="1d")
    spy_df   = normalize_df(flatten_df(spy_df))

    is_trending, regime = check_regime(spy_df)
    spy_price  = spy_df["Close"].iloc[-1]
    spy_ma200  = spy_df["Close"].rolling(200).mean().iloc[-1]

    print(f"  SPY: ${spy_price:.2f} | 200-day MA: ${spy_ma200:.2f}")
    print(f"  Regime: {regime.upper()}")

    if not is_trending:
        print("\n⚠ MARKET IS CHOPPY -- no new entries recommended.")
        print("  Idle capital should be in T-bills, not equities.")
        print("  Orders below shown for reference only.\n")
    else:
        print("  ✓ Regime is trending -- entries are valid.\n")

    # ── Fetch price data for candidates ──────────────────────────────────
    tickers    = candidates["ticker"].tolist()
    print(f"Fetching price data for {len(tickers)} candidates...")
    price_data = provider.get_history_bulk(tickers, period="1y", interval="1d")

    # Fetch betas vs SPY
    print("Computing betas...")
    betas = {}
    for ticker, df in price_data.items():
        df = normalize_df(flatten_df(df))
        price_data[ticker] = df
        try:
            beta = compute_rolling_beta(df, spy_df, window=63)
            betas[ticker] = float(beta.iloc[-1]) if not pd.isna(beta.iloc[-1]) else 1.0
        except Exception:
            betas[ticker] = 1.0

    # ── Generate orders ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATED ORDERS")
    print("Date:", datetime.date.today().strftime("%Y-%m-%d"))
    print("=" * 60)

    orders = []
    for _, row in candidates.iterrows():
        ticker = row["ticker"]
        if ticker not in price_data or price_data[ticker].empty:
            continue

        df    = price_data[ticker]
        price = df["Close"].iloc[-1]
        atr14 = compute_atr14(df)
        beta  = betas.get(ticker, 1.0)
        sar_stop, chand_stop = compute_stop_levels(df)
        stop_level = max(sar_stop, chand_stop)  # tighter of the two
        risk_pct   = (price - stop_level) / price * 100

        size       = compute_position_size(price, atr14, beta)
        shares     = int(size / price)
        actual_size = shares * price

        if shares < 1:
            continue

        orders.append({
            "ticker":         ticker,
            "action":         "BUY",
            "shares":         shares,
            "price":          round(price, 2),
            "position_$":     round(actual_size, 2),
            "pct_portfolio":  round(actual_size / PORTFOLIO_VALUE * 100, 1),
            "beta":           round(beta, 2),
            "atr14":          round(atr14, 2),
            "sar_stop":       round(sar_stop, 2),
            "chandelier_stop": round(chand_stop, 2),
            "stop_level":     round(stop_level, 2),
            "risk_pct":       round(risk_pct, 1),
            "prob_up_5pct":   row["prob_pct"],
            "active_signals": int(row["active_signals"]),
            "weekly_vol":     row["weekly_vol_pct"],
        })

    if not orders:
        print("No valid orders generated (insufficient price data)")
        return

    orders_df = pd.DataFrame(orders)

    gross_deployed = orders_df["position_$"].sum()
    orders_df = cap_orders_to_budget(orders_df)
    total_deployed = orders_df["position_$"].sum()
    total_commissions = orders_df["commission_$"].sum()
    total_cost = orders_df["total_cost_$"].sum()

    if orders_df.empty:
        print("No valid orders remain after applying the deployment cap")
        return

    # Display
    display_cols = ["ticker", "action", "shares", "price",
                    "position_$", "commission_$", "total_cost_$",
                    "pct_portfolio", "stop_level",
                    "risk_pct", "prob_up_5pct", "active_signals"]
    print(orders_df[display_cols].to_string(index=False))

    print(f"\n{'─'*60}")
    print(f"Total positions:    {len(orders_df)}")
    print(f"Raw deployment:    ${gross_deployed:,.0f} "
          f"({gross_deployed/PORTFOLIO_VALUE*100:.1f}% before cap)")
    print(f"Total deployed:    ${total_deployed:,.2f}")
    print(f"Commissions:       ${total_commissions:,.2f}")
    print(f"Total cost:        ${total_cost:,.2f} "
          f"({total_cost/PORTFOLIO_VALUE*100:.1f}% of portfolio)")
    print(f"Remaining idle:    ${PORTFOLIO_VALUE - total_cost:,.2f} "
          f"→ {'SPY' if is_trending else 'T-bills'}")
    print(f"Min position:      ${PORTFOLIO_VALUE * MIN_POSITION_PCT:,.0f} "
          f"({MIN_POSITION_PCT*100:.1f}% of portfolio)")
    print(f"Regime:            {regime.upper()}")

    print("\n" + "=" * 60)
    print("EXIT PLAN (monitor daily)")
    print("=" * 60)
    for _, o in orders_df.iterrows():
        print(f"  {o['ticker']:6s}: stop at ${o['stop_level']:.2f} "
              f"(SAR ${o['sar_stop']:.2f} | "
              f"Chandelier ${o['chandelier_stop']:.2f}) "
              f"| risk {o['risk_pct']:.1f}% from entry")

    print("\n" + "=" * 60)
    print("IBKR INTEGRATION NOTE")
    print("=" * 60)
    print("To place these orders via IBKR:")
    print("  1. Install: pip install ib_async")
    print("  2. Open IB Gateway (paper trading mode to test)")
    print("  3. Replace the print statements above with:")
    print("     from ib_async import IB, MarketOrder")
    print("     ib = IB(); ib.connect('127.0.0.1', 7497, clientId=1)")
    print("     for order in orders:")
    print("         contract = Stock(order['ticker'], 'SMART', 'USD')")
    print("         ib.placeOrder(contract, MarketOrder('BUY', order['shares']))")

    # Save orders
    orders_df.to_csv("orders_today.csv", index=False)
    print(f"\nOrders saved to orders_today.csv")
    print("\n⚠  REMINDER: This is for paper trading / research only.")
    print("   Validate the edge holds out-of-sample before using real capital.")


if __name__ == "__main__":
    main()
