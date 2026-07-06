#!/usr/bin/env python3
"""
AlgoTrader v7.5: Portfolio Construction with Confidence Filter

This script keeps the ML probability model unchanged and adds a true confidence
filter before portfolio construction. It only trades when predictions are strong
enough, caps each stock at 7%, and allows holdings count to be determined by
confidence rather than a fixed small number.

It compares:
  1. ML Baseline: top-probability names with simple capped allocation
  2. ML Optimized: probability-weighted, volatility-targeted, correlation-aware allocation
  3. SPY: buy-and-hold benchmark over the same dates

Idle capital is invested in a cash proxy (default SGOV) instead of earning 0%.

Required first:
  python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2

Then run:
  python run.py construct

Outputs:
  portfolio_construction_report.html
  portfolio_construction_overall.csv
  portfolio_construction_yearly.csv
  portfolio_construction_equity.csv
  portfolio_construction_trades.csv
  portfolio_construction_allocations.csv
  portfolio_construction_equity_curve.png
  portfolio_construction_drawdown.png
  portfolio_construction_yearly_returns.png
  portfolio_construction_exposure.png
"""

import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ----------------------------- User knobs -----------------------------

INITIAL_CAPITAL = 100_000.0
HOLD_DAYS = 5

# Candidate rules
PROB_THRESHOLD = 0.30
CONFIDENCE_DAILY_QUANTILE = 0.95      # only top 5% of daily predictions qualify
CONFIDENCE_MIN_EDGE_OVER_MEDIAN = 0.05 # probability must be at least 5 pct points over daily median
CANDIDATE_POOL = 250                  # broad pool; confidence filter is the real limiter
BASELINE_TOP_N = 10_000               # no meaningful fixed holdings cap
OPTIMIZER_MAX_HOLDINGS = 10_000       # no meaningful fixed holdings cap

# Position / portfolio risk rules
BASELINE_MAX_POSITION_PCT = 0.07
OPTIMIZER_MAX_POSITION_PCT = 0.07
MIN_POSITION_PCT = 0.01
TARGET_PORTFOLIO_VOL = 0.22       # annualized target vol before regime caps
MAX_GROSS_EXPOSURE = 1.00
MIN_GROSS_EXPOSURE = 0.20
MAX_PAIRWISE_CORR = 0.75
CORR_LOOKBACK_DAYS = 63

# Regime / drawdown controls
SPY_VOL_HIGH = 0.28
SPY_VOL_EXTREME = 0.38
DRAWDOWN_CUT_1 = -0.15             # exposure cap 70%
DRAWDOWN_CUT_2 = -0.25             # exposure cap 45%
DRAWDOWN_CUT_3 = -0.35             # exposure cap 25%

# Costs
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00
SLIPPAGE_BPS = 10                  # slightly more conservative than V6 baseline

# Idle-cash / treasury-bill proxy
CASH_YIELD_ENABLED = True
CASH_PROXY = "SGOV"
CASH_FALLBACK_ANNUAL_YIELD = 0.04

ML_PREDICTIONS_FILE = "ml_probability_predictions.csv"
TRAINING_DATA_FILE = "probability_training_dataset.csv"

PERIODS = [
    ("2018 correction", "2018-01-01", "2018-12-31"),
    ("2019 bull market", "2019-01-01", "2019-12-31"),
    ("2020 COVID / recovery", "2020-01-01", "2020-12-31"),
    ("2021 bull / reopening", "2021-01-01", "2021-12-31"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
    ("2023 recovery", "2023-01-01", "2023-12-31"),
    ("2024 AI rally", "2024-01-01", "2024-12-31"),
    ("2025-present", "2025-01-01", None),
]


# ----------------------------- Metrics -----------------------------


def commission(shares: int) -> float:
    return max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE)


def max_drawdown(equity):
    s = pd.Series(equity).astype(float)
    peak = s.cummax()
    dd = s / peak - 1
    return float(dd.min()), dd


def sharpe_ratio(rets):
    r = pd.Series(rets).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return np.sqrt(252) * r.mean() / r.std()


def sortino_ratio(rets):
    r = pd.Series(rets).dropna()
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return np.nan
    return np.sqrt(252) * r.mean() / downside.std()


def cagr(equity, dates):
    e = pd.Series(equity).astype(float)
    d = pd.to_datetime(dates)
    if len(e) < 2:
        return np.nan
    years = (d.iloc[-1] - d.iloc[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1


def metrics_for_equity(g, asset):
    g = g.sort_values("date").copy()
    g["return"] = g["equity"].pct_change()
    mdd, _ = max_drawdown(g["equity"])
    cg = cagr(g["equity"], g["date"])
    return {
        "asset": asset,
        "start": g["date"].iloc[0].date(),
        "end": g["date"].iloc[-1].date(),
        "total_return_pct": (g["equity"].iloc[-1] / g["equity"].iloc[0] - 1) * 100,
        "cagr_pct": cg * 100,
        "volatility_pct": g["return"].std() * np.sqrt(252) * 100,
        "max_drawdown_pct": mdd * 100,
        "sharpe": sharpe_ratio(g["return"]),
        "sortino": sortino_ratio(g["return"]),
        "calmar": cg / abs(mdd) if mdd < 0 else np.nan,
    }


# ----------------------------- Data loading -----------------------------


def require_inputs():
    missing = [f for f in [ML_PREDICTIONS_FILE, TRAINING_DATA_FILE] if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nRun first:\n"
            + "  python run.py train-ml --start 2016-01-01 --model gbdt --min-train-years 2"
        )


def load_data():
    require_inputs()
    ml = pd.read_csv(ML_PREDICTIONS_FILE)
    data = pd.read_csv(TRAINING_DATA_FILE)
    ml["date"] = pd.to_datetime(ml["date"])
    data["date"] = pd.to_datetime(data["date"])

    prob_col = None
    for c in ["ml_prob_up_5pct", "ml_prob", "probability", "pred_prob"]:
        if c in ml.columns:
            prob_col = c
            break
    if prob_col is None:
        raise ValueError("Could not find ML probability column.")

    ml["ml_score"] = pd.to_numeric(ml[prob_col], errors="coerce")
    if ml["ml_score"].max() > 1.5:
        ml["ml_score"] = ml["ml_score"] / 100.0

    needed = [
        "date", "ticker", "close", "forward_return_5d", "vol_21d", "vol_63d", "atr_pct_14d",
        "spy_above_200d", "spy_vol_21d", "ret_5d", "ret_21d", "ret_63d", "dollar_volume_20d"
    ]
    existing = [c for c in needed if c in data.columns]
    merged = ml.merge(data[existing], on=["date", "ticker"], how="left", suffixes=("", "_feat"))

    # Normalize close column if both files had close.
    if "close_feat" in merged.columns and "close" not in merged.columns:
        merged["close"] = merged["close_feat"]
    if "close_feat" in merged.columns:
        merged["close"] = merged["close"].fillna(merged["close_feat"])

    merged = merged.dropna(subset=["date", "ticker", "close", "ml_score", "forward_return_5d"])
    return merged, data


def spy_equity(start, end):
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if spy.empty:
        return pd.DataFrame(columns=["date", "equity", "asset"])
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    spy = spy.reset_index()
    spy.columns = [str(c).lower() for c in spy.columns]
    date_col = "date" if "date" in spy.columns else "datetime"
    spy = spy[[date_col, "close"]].rename(columns={date_col: "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy["equity"] = INITIAL_CAPITAL * spy["close"] / spy["close"].iloc[0]
    spy["asset"] = "SPY"
    return spy[["date", "equity", "asset"]]


def cash_proxy_returns(start, end, rebal_dates):
    """Return a mapping {rebalance_date: cash return over the next HOLD_DAYS trading days}.

    Idle cash is parked in SGOV/BIL-like T-bill proxy. If download fails or the
    proxy lacks enough history, fallback uses a flat annualized cash yield.
    """
    dates = pd.to_datetime(pd.Series(sorted(rebal_dates))).reset_index(drop=True)
    if dates.empty:
        return {}, pd.DataFrame()

    fallback_period_return = (1 + CASH_FALLBACK_ANNUAL_YIELD) ** (HOLD_DAYS / 252) - 1
    ret_map = {pd.Timestamp(d): fallback_period_return for d in dates}

    if not CASH_YIELD_ENABLED:
        return {pd.Timestamp(d): 0.0 for d in dates}, pd.DataFrame()

    try:
        cash = yf.download(CASH_PROXY, start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        cash = pd.DataFrame()

    if cash is None or cash.empty:
        print(f"Cash proxy {CASH_PROXY} unavailable; using fallback {CASH_FALLBACK_ANNUAL_YIELD:.1%} annual yield.")
        return ret_map, pd.DataFrame()

    if isinstance(cash.columns, pd.MultiIndex):
        cash.columns = [c[0] for c in cash.columns]
    cash = cash.reset_index()
    cash.columns = [str(c).lower() for c in cash.columns]
    date_col = "date" if "date" in cash.columns else "datetime"
    cash = cash[[date_col, "close"]].rename(columns={date_col: "date"}).dropna()
    cash["date"] = pd.to_datetime(cash["date"])
    cash = cash.sort_values("date")
    if len(cash) < 2:
        return ret_map, cash

    ser = cash.set_index("date")["close"].astype(float)
    for d in dates:
        d = pd.Timestamp(d)
        # Use nearest available close at/after rebalance date and at/after target date.
        try:
            start_idx = ser.index.searchsorted(d, side="left")
            end_idx = min(start_idx + HOLD_DAYS, len(ser) - 1)
            if start_idx < len(ser) and end_idx > start_idx:
                r = ser.iloc[end_idx] / ser.iloc[start_idx] - 1
                if pd.notna(r):
                    ret_map[d] = float(r)
        except Exception:
            pass
    return ret_map, cash




# ----------------------------- Confidence filter -----------------------------


def confidence_filter(day):
    """Keep only predictions strong enough to justify a trade.

    This is intentionally stricter than simple ranking. A stock must clear all of:
      1. absolute minimum ML probability,
      2. daily top-quantile threshold,
      3. meaningful edge over that day's median prediction.

    If nothing clears the threshold, the strategy stays in cash/SGOV.
    """
    day = day.dropna(subset=["ml_score"]).copy()
    if day.empty:
        return day

    daily_quantile_cutoff = float(day["ml_score"].quantile(CONFIDENCE_DAILY_QUANTILE))
    daily_median = float(day["ml_score"].median())
    threshold = max(
        float(PROB_THRESHOLD),
        daily_quantile_cutoff,
        daily_median + float(CONFIDENCE_MIN_EDGE_OVER_MEDIAN),
    )

    out = day[day["ml_score"] >= threshold].copy()
    if out.empty:
        return out

    out["confidence_threshold"] = threshold
    out["confidence_edge_vs_threshold"] = out["ml_score"] - threshold
    out["confidence_daily_rank_pct"] = out["ml_score"].rank(pct=True)
    return out


def capped_probability_weights(raw, max_weight):
    """Normalize scores while respecting a hard per-position cap.

    If there are too few qualifying stocks to fully deploy capital without
    breaching the cap, the unused capital remains cash/SGOV.
    """
    raw = pd.Series(raw).replace([np.inf, -np.inf], np.nan).dropna()
    raw = raw[raw > 0]
    if raw.empty:
        return {}

    remaining = raw.copy()
    remaining_weight = 1.0
    weights = {}

    while len(remaining) > 0 and remaining_weight > 1e-12:
        trial = remaining / remaining.sum() * remaining_weight
        over = trial[trial > max_weight]
        if over.empty:
            weights.update(trial.to_dict())
            break
        for t in over.index:
            weights[t] = max_weight
        remaining_weight -= max_weight * len(over)
        remaining = remaining.drop(index=over.index)

    # Never renormalize above the cap. If total weight is below 100%, cash stays idle.
    return {t: float(min(w, max_weight)) for t, w in weights.items() if w > 0}

# ----------------------------- Optimizer helpers -----------------------------


def drawdown_exposure_cap(current_equity, peak_equity):
    dd = current_equity / peak_equity - 1 if peak_equity > 0 else 0.0
    if dd <= DRAWDOWN_CUT_3:
        return 0.25
    if dd <= DRAWDOWN_CUT_2:
        return 0.45
    if dd <= DRAWDOWN_CUT_1:
        return 0.70
    return 1.00


def regime_exposure_cap(row):
    cap = 1.00
    spy_above = row.get("spy_above_200d", 1.0)
    spy_vol = row.get("spy_vol_21d", np.nan)
    if pd.notna(spy_above) and spy_above <= 0:
        cap = min(cap, 0.55)
    if pd.notna(spy_vol) and spy_vol >= SPY_VOL_HIGH:
        cap = min(cap, 0.65)
    if pd.notna(spy_vol) and spy_vol >= SPY_VOL_EXTREME:
        cap = min(cap, 0.35)
    return cap


def get_return_matrix(data, dt, tickers, lookback=CORR_LOOKBACK_DAYS):
    sub = data[(data["ticker"].isin(tickers)) & (data["date"] < dt)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("date")
    piv = sub.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    piv = piv.tail(lookback + 1)
    rets = piv.pct_change().dropna(how="all")
    return rets


def estimate_portfolio_vol(weights, candidates, data, dt):
    tickers = list(weights.keys())
    if not tickers:
        return 0.0
    vols = candidates.set_index("ticker").reindex(tickers)["vol_21d"].fillna(candidates["vol_21d"].median()).fillna(0.50).values
    w = np.array([weights[t] for t in tickers], dtype=float)

    rets = get_return_matrix(data, dt, tickers)
    if rets.shape[1] >= 2 and len(rets) > 10:
        corr = rets[tickers].corr().fillna(0).values if all(t in rets.columns for t in tickers) else np.eye(len(tickers))
        corr = np.clip(corr, -0.5, 1.0)
    else:
        corr = np.eye(len(tickers))

    cov = np.outer(vols, vols) * corr
    port_var = float(w.T @ cov @ w)
    return np.sqrt(max(port_var, 0))


def correlation_screen(candidates, data, dt):
    """Greedy selection that avoids highly correlated names."""
    candidates = candidates.sort_values("raw_score", ascending=False).copy()
    selected = []
    rets = get_return_matrix(data, dt, candidates["ticker"].tolist())

    for _, row in candidates.iterrows():
        t = row["ticker"]
        if len(selected) >= OPTIMIZER_MAX_HOLDINGS:
            break
        if not selected or rets.empty or t not in rets.columns:
            selected.append(t)
            continue
        available = [s for s in selected if s in rets.columns]
        if not available:
            selected.append(t)
            continue
        corr = rets[available + [t]].corr()[t].drop(t).abs().max()
        if pd.isna(corr) or corr <= MAX_PAIRWISE_CORR:
            selected.append(t)

    return candidates[candidates["ticker"].isin(selected)].copy()


def optimized_weights(day, data, dt, current_equity, peak_equity):
    candidates = confidence_filter(day)
    candidates = candidates.sort_values("ml_score", ascending=False).head(CANDIDATE_POOL)
    if candidates.empty:
        return {}, 0.0, pd.DataFrame()

    candidates["vol_for_size"] = candidates["vol_21d"].replace([np.inf, -np.inf], np.nan)
    fallback_vol = candidates["vol_for_size"].median()
    if pd.isna(fallback_vol) or fallback_vol <= 0:
        fallback_vol = 0.60
    candidates["vol_for_size"] = candidates["vol_for_size"].fillna(fallback_vol).clip(0.15, 2.50)

    # Probability-weighted, volatility-adjusted score.
    # This intentionally favors high confidence but penalizes very volatile names.
    candidates["edge"] = (candidates["ml_score"] - PROB_THRESHOLD).clip(lower=0.001)
    candidates["raw_score"] = candidates["edge"] / (candidates["vol_for_size"] ** 1.25)

    candidates = correlation_screen(candidates, data, dt)
    if candidates.empty:
        return {}, 0.0, pd.DataFrame()

    raw = candidates.set_index("ticker")["raw_score"]
    weights = capped_probability_weights(raw, OPTIMIZER_MAX_POSITION_PCT)

    # Drop tiny weights, but do not renormalize above the 7% hard cap.
    weights = {t: w for t, w in weights.items() if w >= MIN_POSITION_PCT}
    if sum(weights.values()) <= 0:
        return {}, 0.0, candidates

    # Volatility target and regime/drawdown caps.
    port_vol = estimate_portfolio_vol(weights, candidates, data, dt)
    vol_scale = TARGET_PORTFOLIO_VOL / port_vol if port_vol > 0 else 1.0
    vol_scale = float(np.clip(vol_scale, MIN_GROSS_EXPOSURE, MAX_GROSS_EXPOSURE))

    regime_cap = regime_exposure_cap(candidates.iloc[0])
    dd_cap = drawdown_exposure_cap(current_equity, peak_equity)
    exposure = min(MAX_GROSS_EXPOSURE, vol_scale, regime_cap, dd_cap)
    exposure = max(exposure, MIN_GROSS_EXPOSURE if weights else 0.0)

    weights = {t: w * exposure for t, w in weights.items()}
    return weights, exposure, candidates


# ----------------------------- Simulation -----------------------------


def execute_period(day, weights, equity, asset, dt):
    trades = []
    period_pnl = 0.0
    deployed = 0.0

    indexed = day.set_index("ticker")
    for ticker, weight in weights.items():
        if ticker not in indexed.index:
            continue
        row = indexed.loc[ticker]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        allocation = equity * weight
        if allocation < equity * MIN_POSITION_PCT:
            continue

        entry_price = float(row["close"]) * (1 + SLIPPAGE_BPS / 10000)
        if entry_price <= 0:
            continue
        shares = int((allocation - MIN_COMMISSION) / entry_price)
        while shares > 0:
            buy_commission = commission(shares)
            entry_value = shares * entry_price
            entry_cost = entry_value + buy_commission
            if entry_cost <= allocation:
                break
            shares -= 1
        if shares <= 0:
            continue

        fwd_ret = float(row["forward_return_5d"])
        exit_price = entry_price * (1 + fwd_ret) * (1 - SLIPPAGE_BPS / 10000)
        sell_commission = commission(shares)
        exit_value = shares * exit_price - sell_commission
        pnl = exit_value - entry_cost
        period_pnl += pnl
        deployed += entry_cost

        trades.append({
            "asset": asset,
            "date": dt,
            "ticker": ticker,
            "weight": weight,
            "score": float(row.get("ml_score", np.nan)),
            "shares": shares,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_cost": entry_cost,
            "exit_value": exit_value,
            "commission": buy_commission + sell_commission,
            "forward_return_5d_pct": fwd_ret * 100,
            "net_pnl": pnl,
            "net_return_pct": pnl / entry_cost * 100 if entry_cost else np.nan,
        })

    return period_pnl, deployed, trades


def simulate_baseline(scores, cash_returns=None):
    df = scores.copy()
    df = df.dropna(subset=["date", "ticker", "close", "ml_score", "forward_return_5d"])
    dates = sorted(df["date"].unique())
    rebal_dates = dates[::HOLD_DAYS]

    equity = INITIAL_CAPITAL
    records, trades, allocs = [], [], []
    cash_returns = cash_returns or {}

    for dt in rebal_dates:
        day = df[df["date"] == dt].copy()
        candidates = confidence_filter(day).sort_values("ml_score", ascending=False).head(BASELINE_TOP_N)
        if candidates.empty:
            records.append({"date": dt, "equity": equity, "asset": "ML Baseline"})
            continue

        raw = candidates.set_index("ticker")["ml_score"]
        weights = capped_probability_weights(raw, BASELINE_MAX_POSITION_PCT)

        start_equity = equity
        pnl, deployed, trade_rows = execute_period(candidates, weights, equity, "ML Baseline", dt)
        idle_cash = max(start_equity - deployed, 0.0)
        cash_ret = cash_returns.get(pd.Timestamp(dt), 0.0) if CASH_YIELD_ENABLED else 0.0
        cash_pnl = idle_cash * cash_ret
        equity += pnl + cash_pnl
        trades.extend(trade_rows)
        records.append({"date": dt, "equity": equity, "asset": "ML Baseline"})
        allocs.append({
            "asset": "ML Baseline", "date": dt,
            "gross_exposure": sum(weights.values()),
            "cash_exposure": max(1.0 - sum(weights.values()), 0.0),
            "deployed_pct": deployed / max(start_equity, 1),
            "idle_cash_pct": idle_cash / max(start_equity, 1),
            "cash_period_return_pct": cash_ret * 100,
            "cash_pnl": cash_pnl,
            "holdings": len(trade_rows), "estimated_vol": np.nan
        })

    return pd.DataFrame(records), pd.DataFrame(trades), pd.DataFrame(allocs)


def simulate_optimized(scores, data, cash_returns=None):
    df = scores.copy()
    df = df.dropna(subset=["date", "ticker", "close", "ml_score", "forward_return_5d"])
    dates = sorted(df["date"].unique())
    rebal_dates = dates[::HOLD_DAYS]

    equity = INITIAL_CAPITAL
    peak = equity
    records, trades, allocs = [], [], []
    cash_returns = cash_returns or {}

    for dt in rebal_dates:
        day = df[df["date"] == dt].copy()
        weights, exposure, candidates = optimized_weights(day, data, dt, equity, peak)
        est_vol = estimate_portfolio_vol(weights, candidates, data, dt) if weights else 0.0
        start_equity = equity
        pnl, deployed, trade_rows = execute_period(day, weights, equity, "ML Optimized", dt)
        idle_cash = max(start_equity - deployed, 0.0)
        cash_ret = cash_returns.get(pd.Timestamp(dt), 0.0) if CASH_YIELD_ENABLED else 0.0
        cash_pnl = idle_cash * cash_ret
        equity += pnl + cash_pnl
        peak = max(peak, equity)

        trades.extend(trade_rows)
        records.append({"date": dt, "equity": equity, "asset": "ML Optimized"})
        allocs.append({
            "asset": "ML Optimized",
            "date": dt,
            "gross_exposure": sum(weights.values()),
            "cash_exposure": max(1.0 - sum(weights.values()), 0.0),
            "deployed_pct": deployed / max(start_equity, 1),
            "idle_cash_pct": idle_cash / max(start_equity, 1),
            "cash_period_return_pct": cash_ret * 100,
            "cash_pnl": cash_pnl,
            "holdings": len(trade_rows),
            "estimated_vol": est_vol,
            "drawdown_before_trade_pct": (start_equity / peak - 1) * 100 if peak else 0,
        })

    return pd.DataFrame(records), pd.DataFrame(trades), pd.DataFrame(allocs)


# ----------------------------- Reporting -----------------------------


def overall_summary(equity, trades):
    rows = []
    for asset, g in equity.groupby("asset"):
        row = metrics_for_equity(g, asset)
        if asset != "SPY":
            tg = trades[trades["asset"] == asset]
            wins = tg[tg["net_pnl"] > 0]
            losses = tg[tg["net_pnl"] <= 0]
            row.update({
                "trades": len(tg),
                "win_rate_pct": (tg["net_pnl"] > 0).mean() * 100 if len(tg) else np.nan,
                "avg_trade_pct": tg["net_return_pct"].mean() if len(tg) else np.nan,
                "profit_factor": wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()) if len(losses) and losses["net_pnl"].sum() != 0 else np.nan,
                "total_commissions": tg["commission"].sum() if len(tg) else np.nan,
            })
        else:
            row.update({"trades": 0, "win_rate_pct": np.nan, "avg_trade_pct": np.nan, "profit_factor": np.nan, "total_commissions": np.nan})
        rows.append(row)

    out = pd.DataFrame(rows)

    def diff(a, b, label):
        aa = out[out["asset"] == a]
        bb = out[out["asset"] == b]
        if aa.empty or bb.empty:
            return None
        aa, bb = aa.iloc[0], bb.iloc[0]
        return {
            "asset": label,
            "start": np.nan,
            "end": np.nan,
            "total_return_pct": aa["total_return_pct"] - bb["total_return_pct"],
            "cagr_pct": aa["cagr_pct"] - bb["cagr_pct"],
            "volatility_pct": aa["volatility_pct"] - bb["volatility_pct"],
            "max_drawdown_pct": aa["max_drawdown_pct"] - bb["max_drawdown_pct"],
            "sharpe": aa["sharpe"] - bb["sharpe"],
            "sortino": aa["sortino"] - bb["sortino"],
            "calmar": aa["calmar"] - bb["calmar"],
            "trades": aa.get("trades", np.nan),
            "win_rate_pct": aa.get("win_rate_pct", np.nan),
            "avg_trade_pct": aa.get("avg_trade_pct", np.nan),
            "profit_factor": aa.get("profit_factor", np.nan),
            "total_commissions": aa.get("total_commissions", np.nan),
        }

    extras = [
        diff("ML Optimized", "ML Baseline", "Optimized minus Baseline"),
        diff("ML Optimized", "SPY", "Optimized minus SPY"),
    ]
    extras = [x for x in extras if x is not None]
    if extras:
        out = pd.concat([out, pd.DataFrame(extras)], ignore_index=True)
    return out


def yearly_summary(equity):
    rows = []
    for asset, g in equity.groupby("asset"):
        g = g.sort_values("date").copy()
        g["year"] = g["date"].dt.year
        for year, yg in g.groupby("year"):
            mdd, _ = max_drawdown(yg["equity"])
            rows.append({
                "year": year,
                "asset": asset,
                "return_pct": (yg["equity"].iloc[-1] / yg["equity"].iloc[0] - 1) * 100,
                "max_drawdown_pct": mdd * 100,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pivot = out.pivot(index="year", columns="asset", values="return_pct").reset_index()
    if "ML Optimized" in pivot.columns and "ML Baseline" in pivot.columns:
        pivot["optimized_minus_baseline_pct"] = pivot["ML Optimized"] - pivot["ML Baseline"]
    if "ML Optimized" in pivot.columns and "SPY" in pivot.columns:
        pivot["optimized_minus_spy_pct"] = pivot["ML Optimized"] - pivot["SPY"]
        pivot["optimized_beat_spy"] = pivot["optimized_minus_spy_pct"] > 0
    return pivot


def regime_summary(equity):
    rows = []
    end_default = equity["date"].max()
    for label, start, end in PERIODS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else end_default
        for asset, g in equity.groupby("asset"):
            sub = g[(g["date"] >= s) & (g["date"] <= e)].sort_values("date")
            if len(sub) < 2:
                continue
            row = metrics_for_equity(sub, asset)
            row["period"] = label
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    compact = []
    for period, g in out.groupby("period"):
        row = {"period": period}
        for asset in ["ML Optimized", "ML Baseline", "SPY"]:
            ag = g[g["asset"] == asset]
            if ag.empty:
                continue
            ag = ag.iloc[0]
            prefix = asset.lower().replace(" ", "_")
            row[f"{prefix}_return_pct"] = ag["total_return_pct"]
            row[f"{prefix}_max_drawdown_pct"] = ag["max_drawdown_pct"]
            row[f"{prefix}_sharpe"] = ag["sharpe"]
        if "ml_optimized_return_pct" in row and "ml_baseline_return_pct" in row:
            row["optimized_minus_baseline_pct"] = row["ml_optimized_return_pct"] - row["ml_baseline_return_pct"]
        if "ml_optimized_return_pct" in row and "spy_return_pct" in row:
            row["optimized_minus_spy_pct"] = row["ml_optimized_return_pct"] - row["spy_return_pct"]
        compact.append(row)
    return pd.DataFrame(compact)


def make_charts(equity, yearly, allocs):
    files = []

    plt.figure(figsize=(12, 6))
    for asset, g in equity.groupby("asset"):
        g = g.sort_values("date")
        plt.plot(g["date"], g["equity"], label=asset)
    plt.title("Portfolio Construction: Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    f = "portfolio_construction_equity_curve.png"
    plt.savefig(f, dpi=160)
    plt.close(); files.append(f)

    plt.figure(figsize=(12, 6))
    for asset, g in equity.groupby("asset"):
        g = g.sort_values("date")
        _, dd = max_drawdown(g["equity"])
        plt.plot(g["date"], dd * 100, label=asset)
    plt.title("Portfolio Construction: Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    f = "portfolio_construction_drawdown.png"
    plt.savefig(f, dpi=160)
    plt.close(); files.append(f)

    if not yearly.empty:
        cols = [c for c in ["ML Optimized", "ML Baseline", "SPY"] if c in yearly.columns]
        if cols:
            ax = yearly.set_index("year")[cols].plot(kind="bar", figsize=(12, 6))
            ax.set_title("Portfolio Construction: Yearly Returns")
            ax.set_ylabel("Return (%)")
            plt.tight_layout()
            f = "portfolio_construction_yearly_returns.png"
            plt.savefig(f, dpi=160)
            plt.close(); files.append(f)

    if not allocs.empty:
        plt.figure(figsize=(12, 6))
        for asset, g in allocs.groupby("asset"):
            g = g.sort_values("date")
            plt.plot(pd.to_datetime(g["date"]), g["gross_exposure"] * 100, label=asset)
        plt.title("Gross Exposure Over Time")
        plt.xlabel("Date")
        plt.ylabel("Gross Exposure (%)")
        plt.legend()
        plt.tight_layout()
        f = "portfolio_construction_exposure.png"
        plt.savefig(f, dpi=160)
        plt.close(); files.append(f)

    return files


def make_html(overall, yearly, regimes, alloc_summary, chart_files):
    verdict = []
    opt = overall[overall["asset"] == "ML Optimized"]
    base = overall[overall["asset"] == "ML Baseline"]
    spy = overall[overall["asset"] == "SPY"]
    if not opt.empty and not base.empty:
        opt, base = opt.iloc[0], base.iloc[0]
        verdict.append(f"Optimized CAGR: {opt['cagr_pct']:.2f}% vs Baseline {base['cagr_pct']:.2f}%")
        verdict.append(f"Optimized max drawdown: {opt['max_drawdown_pct']:.2f}% vs Baseline {base['max_drawdown_pct']:.2f}%")
        verdict.append(f"Optimized Sharpe: {opt['sharpe']:.2f} vs Baseline {base['sharpe']:.2f}")
        if opt['max_drawdown_pct'] > base['max_drawdown_pct']:
            verdict.append("Result: optimized portfolio reduced drawdown versus baseline.")
        else:
            verdict.append("Result: optimized portfolio did not reduce drawdown versus baseline; risk rules need tuning.")
    if not opt.empty and not spy.empty:
        sp = spy.iloc[0]
        verdict.append(f"Optimized CAGR minus SPY: {opt['cagr_pct'] - sp['cagr_pct']:.2f} percentage points")

    html = []
    html.append("<html><head><title>AlgoTrader v7 Portfolio Construction</title>")
    html.append("""
<style>
body { font-family: Arial, sans-serif; margin: 40px; color: #222; }
table { border-collapse: collapse; margin: 20px 0; width: 100%; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 7px; text-align: right; }
th { background: #f3f3f3; text-align: center; }
td:first-child, th:first-child { text-align: left; }
h1, h2 { color: #111; }
img { max-width: 100%; margin: 16px 0 32px 0; border: 1px solid #ddd; }
.verdict { white-space: pre-wrap; background: #eef7ee; padding: 16px; border-left: 5px solid #2e7d32; font-family: monospace; }
.note { background: #fff8dc; padding: 12px; border-left: 4px solid #e0b000; margin: 20px 0; }
</style>
""")
    html.append("</head><body>")
    html.append("<h1>AlgoTrader v7 Portfolio Construction Report</h1>")
    html.append(f"<p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html.append("<div class='note'>This report uses the original/simple SPY regime filter, a strict confidence filter, a 7% max position cap, and parks idle cash in a cash proxy instead of assuming 0% return. It is research only.</div>")
    html.append("<h2>Bottom Line</h2>")
    html.append("<div class='verdict'>" + "\n".join(verdict) + "</div>")
    html.append("<h2>Overall Performance</h2>")
    html.append(overall.to_html(index=False, float_format=lambda x: f"{x:,.2f}"))
    html.append("<h2>Year-by-Year Performance</h2>")
    html.append(yearly.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not yearly.empty else "<p>No yearly data.</p>")
    html.append("<h2>Regime Performance</h2>")
    html.append(regimes.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not regimes.empty else "<p>No regime data.</p>")
    html.append("<h2>Allocation Summary</h2>")
    html.append(alloc_summary.to_html(index=False, float_format=lambda x: f"{x:,.2f}") if not alloc_summary.empty else "<p>No allocation data.</p>")
    html.append("<h2>Charts</h2>")
    for f in chart_files:
        html.append(f"<h3>{f}</h3><img src='{f}' />")
    html.append("</body></html>")
    with open("portfolio_construction_report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def main():
    print("=" * 80)
    print("ALGOTRADER V7.5: CONFIDENCE FILTER + 7% MAX POSITION")
    print("ML Baseline vs ML Optimized vs SPY")
    print("=" * 80)

    scores, data = load_data()
    print(f"Loaded {len(scores):,} ML prediction rows")

    all_dates = sorted(scores["date"].dropna().unique())
    rebal_dates = all_dates[::HOLD_DAYS]
    start_for_cash = pd.to_datetime(min(rebal_dates)).strftime("%Y-%m-%d") if rebal_dates else None
    end_for_cash = (pd.to_datetime(max(rebal_dates)) + pd.Timedelta(days=20)).strftime("%Y-%m-%d") if rebal_dates else None
    print(f"Downloading idle-cash proxy {CASH_PROXY}...")
    cash_returns, cash_df = cash_proxy_returns(start_for_cash, end_for_cash, rebal_dates)

    print("Simulating ML baseline...")
    base_eq, base_trades, base_alloc = simulate_baseline(scores, cash_returns)

    print("Simulating ML optimized portfolio...")
    opt_eq, opt_trades, opt_alloc = simulate_optimized(scores, data, cash_returns)

    start = min(base_eq["date"].min(), opt_eq["date"].min())
    end = max(base_eq["date"].max(), opt_eq["date"].max())
    print("Downloading SPY benchmark...")
    spy = spy_equity(start, end)

    # Align to common date range.
    common_start = max(base_eq["date"].min(), opt_eq["date"].min(), spy["date"].min() if not spy.empty else start)
    common_end = min(base_eq["date"].max(), opt_eq["date"].max(), spy["date"].max() if not spy.empty else end)

    equity = pd.concat([base_eq, opt_eq, spy], ignore_index=True)
    equity = equity[(equity["date"] >= common_start) & (equity["date"] <= common_end)].copy()
    trades = pd.concat([base_trades, opt_trades], ignore_index=True)
    trades = trades[(trades["date"] >= common_start) & (trades["date"] <= common_end)].copy()
    allocs = pd.concat([base_alloc, opt_alloc], ignore_index=True)
    if not allocs.empty:
        allocs["date"] = pd.to_datetime(allocs["date"])
        allocs = allocs[(allocs["date"] >= common_start) & (allocs["date"] <= common_end)].copy()

    overall = overall_summary(equity, trades)
    yearly = yearly_summary(equity)
    regimes = regime_summary(equity)

    if not allocs.empty:
        alloc_summary = allocs.groupby("asset").agg(
            avg_gross_exposure_pct=("gross_exposure", lambda x: x.mean() * 100),
            min_gross_exposure_pct=("gross_exposure", lambda x: x.min() * 100),
            max_gross_exposure_pct=("gross_exposure", lambda x: x.max() * 100),
            avg_deployed_pct=("deployed_pct", lambda x: x.mean() * 100),
            avg_idle_cash_pct=("idle_cash_pct", lambda x: x.mean() * 100),
            avg_cash_proxy_return_pct=("cash_period_return_pct", "mean"),
            total_cash_pnl=("cash_pnl", "sum"),
            avg_holdings=("holdings", "mean"),
            avg_estimated_vol_pct=("estimated_vol", lambda x: x.mean() * 100),
        ).reset_index()
    else:
        alloc_summary = pd.DataFrame()

    chart_files = make_charts(equity, yearly, allocs)
    make_html(overall, yearly, regimes, alloc_summary, chart_files)

    overall.to_csv("portfolio_construction_overall.csv", index=False)
    yearly.to_csv("portfolio_construction_yearly.csv", index=False)
    regimes.to_csv("portfolio_construction_regimes.csv", index=False)
    equity.to_csv("portfolio_construction_equity.csv", index=False)
    trades.to_csv("portfolio_construction_trades.csv", index=False)
    allocs.to_csv("portfolio_construction_allocations.csv", index=False)

    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    print(overall.to_string(index=False))
    print("\nSaved:")
    print("  portfolio_construction_report.html")
    print("  portfolio_construction_overall.csv")
    print("  portfolio_construction_yearly.csv")
    print("  portfolio_construction_regimes.csv")
    print("  portfolio_construction_equity.csv")
    print("  portfolio_construction_trades.csv")
    print("  portfolio_construction_allocations.csv")
    for f in chart_files:
        print(f"  {f}")
    print("\nOpen portfolio_construction_report.html for the full report.")


if __name__ == "__main__":
    main()
