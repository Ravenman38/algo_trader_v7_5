"""
Entry and exit signals with MARKET-CAP-TIERED filters.

Different size stocks behave differently and need different thresholds:

  Mega-cap  (>$200B):  SAR gap > 3x ATR, top 10% momentum, skip volume
  Large-cap ($10B-$200B): SAR gap > 2x ATR, top 15% momentum
  Mid-cap   ($2B-$10B):   SAR gap > 1.5x ATR, top 20% momentum (default)
  Small-cap (<$2B):    SAR gap > 2x ATR, top 20% momentum, strict volume

Exit: Parabolic SAR OR Chandelier 2x ATR -- whichever fires first.
No arbitrary time limits.
"""

import pandas as pd
import numpy as np


# ── Market cap tier classification ──────────────────────────────────────────

def get_cap_tier(market_cap):
    if market_cap is None or market_cap <= 0:
        return "mid"
    if market_cap >= 200_000_000_000:
        return "mega"
    if market_cap >= 10_000_000_000:
        return "large"
    if market_cap >= 2_000_000_000:
        return "mid"
    return "small"


CAP_TIER_PARAMS = {
    "mega":  {"sar_gap_atr_multiple": 3.0, "require_volume": False, "momentum_pct_override": 0.10},
    "large": {"sar_gap_atr_multiple": 2.0, "require_volume": True,  "momentum_pct_override": 0.15},
    "mid":   {"sar_gap_atr_multiple": 1.5, "require_volume": True,  "momentum_pct_override": None},
    "small": {"sar_gap_atr_multiple": 2.0, "require_volume": True,  "momentum_pct_override": None},
}


# ── Parabolic SAR ───────────────────────────────────────────────────────────

def compute_parabolic_sar(df, af_start=0.02, af_step=0.02, af_max=0.20):
    high = df["High"].values
    low  = df["Low"].values
    n    = len(df)
    sar  = np.zeros(n)
    ep   = np.zeros(n)
    af   = np.zeros(n)
    bull = np.ones(n, dtype=bool)
    sar[0] = low[0]; ep[0] = high[0]; af[0] = af_start

    for i in range(1, n):
        if bull[i-1]:
            new_sar = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            new_sar = min(new_sar, low[i-1], low[max(i-2,0)])
            if low[i] < new_sar:
                bull[i]=False; sar[i]=ep[i-1]; ep[i]=low[i]; af[i]=af_start
            else:
                bull[i]=True; sar[i]=new_sar
                if high[i] > ep[i-1]:
                    ep[i]=high[i]; af[i]=min(af[i-1]+af_step, af_max)
                else:
                    ep[i]=ep[i-1]; af[i]=af[i-1]
        else:
            new_sar = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            new_sar = max(new_sar, high[i-1], high[max(i-2,0)])
            if high[i] > new_sar:
                bull[i]=True; sar[i]=ep[i-1]; ep[i]=high[i]; af[i]=af_start
            else:
                bull[i]=False; sar[i]=new_sar
                if low[i] < ep[i-1]:
                    ep[i]=low[i]; af[i]=min(af[i-1]+af_step, af_max)
                else:
                    ep[i]=ep[i-1]; af[i]=af[i-1]

    return pd.Series(sar, index=df.index, name="sar")


# ── Chandelier Exit ─────────────────────────────────────────────────────────

def compute_chandelier_exit(df, atr_period=22, multiplier=2.0):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    return df["Close"].rolling(atr_period).max() - multiplier * atr


# ── MACD ────────────────────────────────────────────────────────────────────

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return pd.DataFrame({
        "macd": macd_line,
        "signal_line": macd_line.ewm(span=signal, adjust=False).mean(),
    }, index=close.index)


# ── Entry indicators ────────────────────────────────────────────────────────

def compute_entry_indicators(df, market_cap=None):
    tier   = get_cap_tier(market_cap)
    params = CAP_TIER_PARAMS[tier]

    out = pd.DataFrame(index=df.index)
    out["close"]  = df["Close"]
    out["volume"] = df["Volume"]
    out["cap_tier"] = tier

    macd_df     = compute_macd(df["Close"])
    cross_above = (
        (macd_df["macd"] > macd_df["signal_line"]) &
        (macd_df["macd"].shift(1) <= macd_df["signal_line"].shift(1))
    )
    out["macd_fresh_cross"] = (
        cross_above | cross_above.shift(1) | cross_above.shift(2)
    ).fillna(False)

    out["above_ma50"] = df["Close"] > df["Close"].rolling(50).mean()

    avg_vol = df["Volume"].rolling(20).mean()
    out["volume_confirmed"] = (df["Volume"] > avg_vol) if params["require_volume"] else True

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()

    sar = compute_parabolic_sar(df)
    out["sar"]          = sar
    out["atr14"]        = atr14
    out["sar_gap_pts"]  = df["Close"] - sar
    out["sar_gap_ok"]   = out["sar_gap_pts"] > (params["sar_gap_atr_multiple"] * atr14)
    out["momentum_pct_override"] = params["momentum_pct_override"]

    return out


def check_entry(indicators, date):
    if date not in indicators.index:
        return False
    row = indicators.loc[date]
    return (
        bool(row.get("macd_fresh_cross", False)) and
        bool(row.get("above_ma50",       False)) and
        bool(row.get("volume_confirmed",  False)) and
        bool(row.get("sar_gap_ok",        False))
    )


def get_momentum_threshold(indicators, date, global_threshold):
    if date not in indicators.index:
        return global_threshold
    override = indicators.loc[date, "momentum_pct_override"]
    if override is not None and not pd.isna(float(override) if override is not None else float('nan')):
        return float(override)
    return global_threshold


# ── Exit indicators ─────────────────────────────────────────────────────────

def compute_exit_indicators(df):
    out = pd.DataFrame(index=df.index)
    out["close"]      = df["Close"]
    out["sar"]        = compute_parabolic_sar(df)
    out["chandelier"] = compute_chandelier_exit(df)
    return out


def find_exit_date(exit_ind, df, entry_date, entry_price):
    if entry_date not in df.index:
        return entry_date, "no_data", entry_price
    entry_idx = df.index.get_loc(entry_date)
    for i in range(1, len(df) - entry_idx):
        idx = entry_idx + i
        if idx >= len(df):
            break
        date  = df.index[idx]
        close = df["Close"].iloc[idx]
        if date not in exit_ind.index:
            continue
        sar        = exit_ind.loc[date, "sar"]
        chandelier = exit_ind.loc[date, "chandelier"]
        if close < sar:
            return date, "sar_exit", close
        if not pd.isna(chandelier) and close < chandelier:
            return date, "chandelier_exit", close
    last_idx = len(df) - 1
    return df.index[last_idx], "end_of_data", df["Close"].iloc[last_idx]
