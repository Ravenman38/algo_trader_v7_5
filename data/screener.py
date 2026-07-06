"""
Two-stage screening to avoid downloading data for the full universe.

Stage 1 -- Pre-filter ticker list (no downloads needed):
  Remove obvious non-stocks from the raw ticker list.
  Cuts ~6,900 tickers to ~2,500 genuine common stocks instantly.

Stage 2 -- Parallel lightweight momentum screen:
  Downloads 15 months of data for all candidates in parallel threads,
  checks quality + momentum criteria. Much faster than sequential.
"""

import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Stage 1: ticker pre-filter ───────────────────────────────────────────────

_DERIVATIVE_SUFFIXES = re.compile(
    r'(W|WS|WW|U|UT|R|RT|P|PR|L|Z|V|Q|A|B|C|D|E|F|G|H|I|J|K|N|O|S|T|X|Y)$',
    re.IGNORECASE
)


def is_common_stock(ticker: str) -> bool:
    t = ticker.strip().upper()
    if len(t) > 5:
        return False
    if len(t) >= 4 and _DERIVATIVE_SUFFIXES.search(t):
        return False
    if t.count('.') > 1:
        return False
    return True


def prefilter_tickers(tickers: list[str]) -> list[str]:
    before   = len(tickers)
    filtered = [t for t in tickers if is_common_stock(t)]
    print(f"[screener] Pre-filter: {before} -> {len(filtered)} tickers "
          f"({before - len(filtered)} removed as non-common-stock)")
    return filtered


# ── Stage 2: parallel lightweight screen ─────────────────────────────────────

def _check_single_ticker(ticker: str, df: pd.DataFrame,
                          min_dollar_volume: float = 5_000_000,
                          min_price: float = 1.0) -> bool:
    """
    Check quality criteria for one ticker given its pre-downloaded DataFrame.
    Returns True if ticker passes, False otherwise.
    """
    try:
        if df is None or df.empty or len(df) < 220:
            return False

        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()

        recent   = df.tail(20)
        avg_price = recent["Close"].mean()
        avg_dv    = (recent["Close"] * recent["Volume"]).mean()

        if avg_price < min_price or avg_dv < min_dollar_volume:
            return False

        high_52w = df["Close"].tail(252).max()
        if df["Close"].iloc[-1] < high_52w * 0.70:
            return False

        ma200 = df["Close"].rolling(200).mean()
        if pd.isna(ma200.iloc[-1]) or pd.isna(ma200.iloc[-21]):
            return False
        if ma200.iloc[-1] <= ma200.iloc[-21]:
            return False

        if len(df) >= 63:
            ret_3m = df["Close"].iloc[-1] / df["Close"].iloc[-63] - 1
            if ret_3m <= 0:
                return False

        return True
    except Exception:
        return False


def quick_screen_ticker(ticker: str, provider,
                         min_dollar_volume: float = 5_000_000,
                         min_price: float = 1.0,
                         lookback_days: int = 300) -> pd.DataFrame | None:
    """Single-ticker screen -- kept for backward compatibility."""
    try:
        df = provider.get_history(ticker, period="15mo", interval="1d")
        if _check_single_ticker(ticker, df, min_dollar_volume, min_price):
            return df
        return None
    except Exception:
        return None


def parallel_screen(tickers: list[str], provider,
                    min_dollar_volume: float = 5_000_000,
                    min_price: float = 1.0,
                    batch_size: int = 200,
                    max_workers: int = 8) -> list[str]:
    """
    Screen all candidates in parallel using bulk downloads + thread pool.

    Strategy:
      1. Download 15 months of data for all tickers in batches using
         yfinance's fast bulk download (much faster than one-by-one)
      2. Check quality criteria for each ticker in parallel threads
      3. Return list of tickers that pass

    max_workers: number of parallel threads for quality checks.
    batch_size: number of tickers per bulk download call.
    """
    passing = []
    total   = len(tickers)
    checked = 0

    for batch_start in range(0, total, batch_size):
        batch = tickers[batch_start: batch_start + batch_size]

        # Bulk download for the whole batch at once
        try:
            bulk_data = provider.get_history_bulk(
                batch, period="15mo", interval="1d"
            )
        except Exception as e:
            print(f"  [batch] Download error: {e}")
            bulk_data = {}

        # Parallel quality checks across the batch
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _check_single_ticker, t,
                    bulk_data.get(t, pd.DataFrame()),
                    min_dollar_volume, min_price
                ): t
                for t in batch
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    if future.result():
                        passing.append(ticker)
                except Exception:
                    pass

        checked += len(batch)
        if checked % 400 == 0 or checked >= total:
            print(f"  [{checked}/{total}] {len(passing)} passing so far...")

    print(f"[screener] {len(passing)} tickers passed parallel screen")
    return passing


def two_stage_screen(tickers, provider, market_caps,
                     min_market_cap=300_000_000,
                     max_market_cap=2_000_000_000,
                     min_dollar_volume=5_000_000,
                     batch_size=200, max_workers=8):
    candidates = prefilter_tickers(tickers)
    if market_caps:
        candidates = [
            t for t in candidates
            if min_market_cap <= market_caps.get(t, 0) <= max_market_cap
        ]
        print(f"[screener] Market cap filter: {len(candidates)} candidates remain")
    return parallel_screen(candidates, provider,
                            min_dollar_volume=min_dollar_volume,
                            batch_size=batch_size,
                            max_workers=max_workers)
