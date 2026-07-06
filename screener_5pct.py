"""
5% Weekly Probability Screener
================================
Focuses ONLY on stocks where a 5% weekly move is realistically possible.

Universe criteria (applied before any signal analysis):
  - Weekly volatility > 3% (minimum for ~10% probability of 5% move)
  - Market cap $300M - $5B (small/mid cap sweet spot)
  - Average daily dollar volume > $5M (liquid enough to trade)
  - Listed for at least 1 year (sufficient price history)
  - Sectors: Technology, Healthcare/Biotech, Energy, Industrials
    (naturally higher volatility; excludes utilities, staples, financials)

For each qualifying stock, estimates P(up 5% in 1 week) using:
  1. Actual weekly volatility from recent price history
  2. Signal-adjusted expected weekly return

Signals that adjust the expected return upward:
  - Momentum acceleration      +0.3% expected weekly return
  - Fresh SAR gap (>3%)        +0.2%
  - Volume surge (>1.5x avg)   +0.2%
  - MACD fresh crossover       +0.2%
  - Momentum top 20%           +0.4%

Backtest validation: tests whether top-probability predictions actually
outperform bottom-probability ones historically. This is the real edge test.
"""

import sys, os, random, datetime
import pandas as pd
import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.provider import YFinanceProvider
from data.universe import get_universe, get_market_caps
from data.screener import prefilter_tickers
from signals.entry_exit import compute_parabolic_sar
from signals.entry_exit import compute_macd as _compute_macd


# ── Universe parameters ────────────────────────────────────────────────────────

MIN_WEEKLY_VOL   = 0.03    # 3% minimum weekly volatility
MAX_WEEKLY_VOL   = 0.20    # 20% cap -- above this is noise/binary-event driven
MIN_MARKET_CAP   = 300_000_000
MAX_MARKET_CAP   = 10_000_000_000  # widened to $10B to get more qualifying stocks
MIN_AVG_DOLLAR_VOL = 5_000_000
MIN_HISTORY_DAYS = 252     # at least 1 year listed
SMOKE_LIMIT      = 900

# High-volatility sectors (yfinance sector strings)
HIGH_VOL_SECTORS = {
    "Technology", "Healthcare", "Energy",
    "Industrials", "Communication Services",
    "Consumer Cyclical", "Basic Materials"
}

# ── Signal weights ─────────────────────────────────────────────────────────────

SIGNAL_ADJUSTMENTS = {
    "momentum_top20":  0.004,
    "acceleration":    0.003,
    "sar_gap_fresh":   0.002,
    "volume_surge":    0.002,
    "macd_cross":      0.002,
}

TARGET_GAIN    = 0.05
HOLDING_DAYS   = 5


# ── Volatility-based universe filter ──────────────────────────────────────────

def _flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance bulk download."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def compute_weekly_vol(df: pd.DataFrame, window: int = 52) -> float:
    """Weekly volatility from daily returns, annualised to weekly."""
    if df is None or len(df) < 30:
        return 0.0
    daily_ret = df["Close"].pct_change().dropna().tail(window * 5)
    return float(daily_ret.std() * np.sqrt(5))


def passes_volatility_filter(df: pd.DataFrame) -> tuple[bool, float]:
    """Returns (passes, weekly_vol)."""
    wv = compute_weekly_vol(df)
    return (MIN_WEEKLY_VOL <= wv <= MAX_WEEKLY_VOL), wv


def get_sector(ticker: str, provider) -> str:
    """Fetch sector for a ticker via yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        # fast_info doesn't have sector; use full info
        full = yf.Ticker(ticker).info
        return full.get("sector", "Unknown")
    except Exception:
        return "Unknown"


def build_volatility_universe(tickers: list[str],
                               provider,
                               batch_size: int = 200,
                               max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """
    Download recent price history for all candidates and filter to
    those with weekly volatility in the target range.

    Returns {ticker: DataFrame} for qualifying stocks only.
    """
    print(f"[universe] Downloading 1-year history for {len(tickers)} candidates...")
    qualifying = {}
    total = len(tickers)
    checked = 0

    for batch_start in range(0, total, batch_size):
        batch = tickers[batch_start: batch_start + batch_size]
        try:
            bulk = provider.get_history_bulk(batch, period="2y", interval="1d")
        except Exception:
            bulk = {}

        for ticker, df in bulk.items():
            if df is None or df.empty:
                continue
            # Normalize index
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()

            if len(df) < MIN_HISTORY_DAYS:
                continue

            # Liquidity check
            recent = df.tail(20)
            avg_dv = (recent["Close"] * recent["Volume"]).mean()
            if avg_dv < MIN_AVG_DOLLAR_VOL:
                continue

            # Volatility filter
            passes, wv = passes_volatility_filter(df)
            if not passes:
                continue

            qualifying[ticker] = df

        checked += len(batch)
        if checked % 400 == 0 or checked >= total:
            print(f"  [{checked}/{total}] {len(qualifying)} qualifying so far...")

    print(f"[universe] {len(qualifying)} stocks pass volatility filter "
          f"({MIN_WEEKLY_VOL*100:.0f}%-{MAX_WEEKLY_VOL*100:.0f}% weekly vol)")
    return qualifying


# ── Signal detection ───────────────────────────────────────────────────────────

def compute_3m_momentum(df: pd.DataFrame) -> float:
    df = _flatten_df(df)
    """
    3-month (63-day) trailing return.
    Much more relevant than 12-1 momentum for 1-week predictions --
    captures what the stock has been doing recently, not a year ago.
    """
    if len(df) < 65:
        return float("nan")
    return float(df["Close"].iloc[-1] / df["Close"].iloc[-63] - 1)


def compute_3m_acceleration(df: pd.DataFrame) -> bool:
    df = _flatten_df(df)
    """
    Short-term acceleration: 1-month return > 3-month annualised return.
    Is the stock speeding up recently (last month vs last 3 months)?
    """
    if len(df) < 65:
        return False
    ret_1m = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1
    ret_3m = df["Close"].iloc[-1] / df["Close"].iloc[-63] - 1
    # Annualise 1-month to same scale as 3-month
    ann_1m = (1 + ret_1m) ** 3 - 1   # 3 periods per quarter
    return bool(ann_1m > ret_3m)


def detect_signals(df: pd.DataFrame, ticker: str,
                    top20_tickers: set) -> dict:
    df      = _flatten_df(df)
    signals = {}
    close   = df["Close"]

    signals["momentum_top20"] = ticker in top20_tickers
    signals["acceleration"]   = compute_3m_acceleration(df)

    try:
        sar = compute_parabolic_sar(df)
        gap = (close.iloc[-1] - sar.iloc[-1]) / close.iloc[-1]
        signals["sar_gap_fresh"] = bool(gap > 0.03)
        signals["sar_gap_pct"]   = round(float(gap * 100), 2)
    except Exception:
        signals["sar_gap_fresh"] = False
        signals["sar_gap_pct"]   = 0.0

    if "Volume" in df.columns and len(df) >= 20:
        avg_vol = df["Volume"].tail(20).mean()
        cur_vol = df["Volume"].iloc[-1]
        signals["volume_surge"] = bool(cur_vol > avg_vol * 1.5)
        signals["volume_ratio"] = round(float(cur_vol / avg_vol), 2) if avg_vol > 0 else 1.0
    else:
        signals["volume_surge"] = False
        signals["volume_ratio"] = 1.0

    if len(df) >= 35:
        macd_df = _compute_macd(df["Close"])
        cross_above = (
            (macd_df["macd"] > macd_df["signal_line"]) &
            (macd_df["macd"].shift(1) <= macd_df["signal_line"].shift(1))
        )
        signals["macd_cross"] = bool(
            cross_above.iloc[-1] or
            cross_above.iloc[-2] or
            cross_above.iloc[-3]
        )
    else:
        signals["macd_cross"] = False

    return signals


def compute_signal_boost(signals: dict) -> float:
    return sum(
        adj for name, adj in SIGNAL_ADJUSTMENTS.items()
        if signals.get(name, False)
    )


# ── Probability calculation ────────────────────────────────────────────────────

def compute_prob_5pct(df: pd.DataFrame, signal_boost: float = 0.0) -> dict | None:
    df        = _flatten_df(df)
    daily_ret = df["Close"].pct_change().dropna()
    if len(daily_ret) < 20:
        return None

    recent   = daily_ret.tail(60)
    σ_daily  = recent.std()
    μ_daily  = recent.mean()
    if σ_daily <= 0 or pd.isna(σ_daily):
        return None

    μ_weekly  = μ_daily * HOLDING_DAYS + signal_boost
    σ_weekly  = σ_daily * np.sqrt(HOLDING_DAYS)
    log_target = np.log(1 + TARGET_GAIN)
    μ_log      = μ_weekly - 0.5 * σ_weekly**2
    z_score    = (log_target - μ_log) / σ_weekly
    prob       = float(1 - stats.norm.cdf(z_score))

    return {
        "prob_up_5pct":   round(prob, 4),
        "weekly_vol_pct": round(σ_weekly * 100, 2),
        "μ_weekly_adj":   round(μ_weekly * 100, 3),
        "signal_boost":   round(signal_boost * 100, 3),
        "z_score":        round(z_score, 3),
    }


# ── Main screener ──────────────────────────────────────────────────────────────

def run_screener(price_data: dict, market_caps: dict,
                  top_n: int = 30) -> pd.DataFrame:
    # Cross-sectional 3-month momentum ranking (not 12-month)
    mom_scores = {}
    for ticker, df in price_data.items():
        try:
            if len(df) < 65:
                continue
            score = compute_3m_momentum(df)
            if score is not None and not pd.isna(score):
                mom_scores[ticker] = float(score)
        except Exception:
            continue

    threshold = np.percentile(list(mom_scores.values()), 80) if mom_scores else 0
    top20     = {t for t, s in mom_scores.items() if s >= threshold}

    rows = []
    skipped = {"no_mom": 0, "no_prob": 0, "error": 0}
    for ticker, df in price_data.items():
        try:
            wv              = compute_weekly_vol(df)
            signals         = detect_signals(df, ticker, top20)
            boost           = compute_signal_boost(signals)
            prob_info       = compute_prob_5pct(df, boost)
            if prob_info is None:
                skipped["no_prob"] += 1
                continue

            active = sum(1 for k in SIGNAL_ADJUSTMENTS if signals.get(k))

            rows.append({
                "ticker":           ticker,
                "price":            round(df["Close"].iloc[-1], 2),
                "market_cap_B":     round(market_caps.get(ticker, 0) / 1e9, 2),
                "weekly_vol_pct":   prob_info["weekly_vol_pct"],
                "prob_up_5pct":     prob_info["prob_up_5pct"],
                "prob_pct":         f"{prob_info['prob_up_5pct']*100:.1f}%",
                "active_signals":   active,
                "signal_boost_pct": prob_info["signal_boost"],
                "momentum_top20":   signals["momentum_top20"],
                "acceleration":     signals["acceleration"],
                "sar_gap_fresh":    signals["sar_gap_fresh"],
                "sar_gap_pct":      signals.get("sar_gap_pct", 0),
                "volume_surge":     signals["volume_surge"],
                "volume_ratio":     signals.get("volume_ratio", 1.0),
                "macd_cross":       signals["macd_cross"],
                "adj_weekly_ret":   prob_info["μ_weekly_adj"],
                "z_score":          prob_info["z_score"],
            })
        except Exception as e:
            skipped["error"] += 1
            if skipped["error"] <= 3:
                import traceback
                print(f"  ERROR on {ticker}: {e}")
                traceback.print_exc()
            continue

    print(f"[screener] {len(rows)} stocks scored, skipped: {skipped}")
    df_out = pd.DataFrame(rows)
    if df_out.empty or "prob_up_5pct" not in df_out.columns:
        print(f"[screener] No rows generated -- check universe size")
        return df_out
    return (df_out
            .sort_values("prob_up_5pct", ascending=False)
            .head(top_n)
            .reset_index(drop=True))


# ── Edge validation backtest ───────────────────────────────────────────────────

def backtest_predictions(price_data: dict, market_caps: dict,
                          lookback_months: int = 24,
                          rescreen_days: int = 5) -> pd.DataFrame:
    print("[backtest] Walk-forward edge validation (last 24 months)...")

    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    cutoff    = pd.Timestamp(all_dates[-1]) - pd.DateOffset(months=lookback_months)
    screen_dates = [d for d in all_dates if pd.Timestamp(d) >= cutoff][::rescreen_days]

    records = []
    for screen_date in screen_dates:
        hist = {t: df[df.index <= screen_date]
                for t, df in price_data.items()
                if len(df[df.index <= screen_date]) >= 60}
        if len(hist) < 10:
            continue

        try:
            # Momentum ranking on historical data
            mom = {}
            for t, df in hist.items():
                try:
                    if len(df) < 65:
                        continue
                    score = compute_3m_momentum(df)
                    if score is not None and not pd.isna(score):
                        mom[t] = float(score)
                except Exception:
                    continue
            thresh = np.percentile(list(mom.values()), 80) if mom else 0
            top20  = {t for t, s in mom.items() if s >= thresh}

            rows = []
            for ticker, df in hist.items():
                wv        = compute_weekly_vol(df)
                if not (MIN_WEEKLY_VOL <= wv <= MAX_WEEKLY_VOL):
                    continue
                signals   = detect_signals(df, ticker, top20)
                boost     = compute_signal_boost(signals)
                prob_info = compute_prob_5pct(df, boost)
                if prob_info:
                    rows.append({
                        "ticker":       ticker,
                        "prob_up_5pct": prob_info["prob_up_5pct"],
                    })

            if len(rows) < 10:
                continue

            ranked  = pd.DataFrame(rows).sort_values("prob_up_5pct", ascending=False)
            n       = len(ranked)
            q       = max(n // 5, 1)
            top_q   = set(ranked.head(q)["ticker"])
            bot_q   = set(ranked.tail(q)["ticker"])

            idx      = all_dates.index(screen_date)
            exit_idx = min(idx + rescreen_days, len(all_dates) - 1)
            exit_dt  = all_dates[exit_idx]

            for ticker in top_q | bot_q:
                df = price_data[ticker]
                if screen_date not in df.index or exit_dt not in df.index:
                    continue
                ep  = df.loc[screen_date, "Close"]
                xp  = df.loc[exit_dt,     "Close"]
                ret = (xp - ep) / ep
                records.append({
                    "screen_date":      screen_date,
                    "ticker":           ticker,
                    "quintile":         "top" if ticker in top_q else "bottom",
                    "actual_5d_return": ret,
                    "up_5pct":          ret >= 0.05,
                })
        except Exception:
            continue

    return pd.DataFrame(records)


def summarize_backtest(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "No backtest results"}
    from scipy import stats as sc
    top = df[df["quintile"] == "top"]
    bot = df[df["quintile"] == "bottom"]

    top_hit  = top["up_5pct"].mean() if len(top) else 0
    bot_hit  = bot["up_5pct"].mean() if len(bot) else 0
    top_ret  = top["actual_5d_return"].mean() if len(top) else 0
    bot_ret  = bot["actual_5d_return"].mean() if len(bot) else 0

    t_stat, p_val = sc.ttest_ind(
        top["actual_5d_return"], bot["actual_5d_return"]
    ) if len(top) > 1 and len(bot) > 1 else (0, 1)

    has_edge = bool(p_val < 0.05 and top_ret > bot_ret)
    return {
        "total_predictions":             len(df),
        "top_quintile_observations":     len(top),
        "bottom_quintile_observations":  len(bot),
        "top_quintile_hit_rate":         round(float(top_hit), 4),
        "bottom_quintile_hit_rate":      round(float(bot_hit), 4),
        "hit_rate_edge":                 round(float(top_hit - bot_hit), 4),
        "top_quintile_avg_5d_return":    round(float(top_ret), 5),
        "bottom_quintile_avg_5d_return": round(float(bot_ret), 5),
        "return_edge":                   round(float(top_ret - bot_ret), 5),
        "t_stat":                        round(float(t_stat), 3),
        "p_value":                       round(float(p_val), 4),
        "significant_at_5pct":           bool(p_val < 0.05),
        "has_edge":                      has_edge,
        "interpretation": (
            "EDGE CONFIRMED: top-probability stocks significantly outperform."
            if has_edge else
            "NO EDGE DETECTED: signals do not significantly predict 5% moves."
        ),
    }


# ── Runner ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("5% WEEKLY PROBABILITY SCREENER")
    print(f"Universe: stocks with {MIN_WEEKLY_VOL*100:.0f}%-{MAX_WEEKLY_VOL*100:.0f}% "
          f"weekly vol, ${MIN_MARKET_CAP/1e9:.1f}B-${MAX_MARKET_CAP/1e9:.1f}B market cap")
    print("=" * 60)

    provider = YFinanceProvider()
    tickers  = get_universe(include_midcap=True, mode="broad")
    if len(tickers) > SMOKE_LIMIT:
        random.seed(42)
        tickers = random.sample(tickers, SMOKE_LIMIT)

    print(f"\n[1/3] Pre-filtering {len(tickers)} tickers...")
    candidates = prefilter_tickers(tickers)
    print(f"  {len(candidates)} common stocks after string filter")

    print(f"\n[2/3] Building volatility-filtered universe...")
    price_data = build_volatility_universe(candidates, provider)

    if len(price_data) < 10:
        print("Universe too small -- try increasing SMOKE_LIMIT or widening filters")
        return

    print(f"\n[3/3] Fetching market caps and applying size filter...")
    market_caps = get_market_caps(list(price_data.keys()))
    price_data  = {
        t: df for t, df in price_data.items()
        if MIN_MARKET_CAP <= market_caps.get(t, 0) <= MAX_MARKET_CAP
    }
    print(f"  {len(price_data)} stocks in final universe")
    if len(price_data) < 5:
        print("  Too few stocks. Widening market cap range...")
        price_data = {
            t: df for t, df in price_data.items()
            if market_caps.get(t, 0) >= MIN_MARKET_CAP
        }
        print(f"  {len(price_data)} stocks after widening")

    # Normalize all indexes (belt and braces)
    for t in list(price_data.keys()):
        df = price_data[t]
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        price_data[t] = df

    # Show volatility distribution
    vols = [compute_weekly_vol(df) for df in price_data.values()]
    print(f"\n  Weekly volatility distribution:")
    print(f"    Median: {np.median(vols)*100:.1f}%")
    print(f"    Mean:   {np.mean(vols)*100:.1f}%")
    print(f"    Min:    {np.min(vols)*100:.1f}%")
    print(f"    Max:    {np.max(vols)*100:.1f}%")

    # ── TODAY'S SCREEN ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TODAY'S TOP 30 BY P(UP 5% IN 1 WEEK)")
    print("=" * 60)

    top30 = run_screener(price_data, market_caps, top_n=30)
    if top30.empty:
        print("No results")
        return

    print(top30[["ticker", "price", "weekly_vol_pct", "prob_pct",
                  "active_signals", "momentum_top20", "acceleration",
                  "sar_gap_fresh", "volume_surge", "macd_cross"]].to_string())

    base_prob = float(1 - stats.norm.cdf(
        (np.log(1.05) - np.mean(vols)**2 / 2) / np.mean(vols)
    ))
    print(f"\n  Base P(up 5%) without signals (median vol stock): "
          f"{base_prob*100:.1f}%")
    print(f"  Top stock: {top30.iloc[0]['ticker']} "
          f"at {top30.iloc[0]['prob_pct']}")
    print(f"  Average (top 30): {top30['prob_up_5pct'].mean()*100:.1f}%")

    # ── EDGE VALIDATION ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EDGE VALIDATION: DO HIGH-PROB PREDICTIONS OUTPERFORM?")
    print("=" * 60)

    results = backtest_predictions(price_data, market_caps)
    summary = summarize_backtest(results)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    top30.to_csv("screener_results.csv", index=False)
    if not results.empty:
        results.to_csv("backtest_predictions.csv", index=False)
    print("\nSaved: screener_results.csv, backtest_predictions.csv")


if __name__ == "__main__":
    main()
