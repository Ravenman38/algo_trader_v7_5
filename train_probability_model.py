"""
Train Probability Model
=======================

Builds a historical stock/day dataset and trains a walk-forward model to estimate:

    P(stock gains >= 5% over the next 5 trading days)

This is intended to replace / benchmark the current heuristic probability score.
It uses only information available up to each decision date and evaluates on
future periods using walk-forward validation.

Outputs:
  - ml_probability_predictions.csv
  - probability_model_summary.csv
  - probability_model_feature_importance.csv
  - probability_model_latest_scores.csv
  - probability_model.pkl

Run:
  python train_probability_model.py

Optional:
  python train_probability_model.py --start 2018-01-01 --max-tickers 900
  python train_probability_model.py --model logistic
  python train_probability_model.py --model gbdt
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception as exc:
    raise SystemExit("Missing yfinance. Run: pip install yfinance") from exc

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception as exc:
    raise SystemExit(
        "Missing scikit-learn/joblib. Run: pip install scikit-learn joblib"
    ) from exc

# Allow imports from the project root when run from Colab/GitHub repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data.universe import get_universe
    from data.screener import prefilter_tickers
except Exception:
    get_universe = None
    prefilter_tickers = None


# ----------------------------- Configuration -----------------------------

TARGET_GAIN = 0.07          # label = 1 if forward 5-day return >= 5%
HOLDING_DAYS = 5
START_DATE = "2018-01-01"
MIN_TRAIN_YEARS = 2
MAX_TICKERS = 900
UNIVERSE_MODE = "sp"
CAP_MODE = "none"  # none or proxy
MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 10_000_000_000
RANDOM_STATE = 42
CACHE_DIR = "data_cache"

FEATURE_COLUMNS = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d",
    "vol_10d", "vol_21d", "vol_63d",
    "atr_pct_14d", "rsi_14d",
    "macd_hist", "macd_signal_gap",
    "volume_ratio_20d", "dollar_volume_20d",
    "dist_ma_20d", "dist_ma_50d", "dist_ma_200d",
    "dist_52w_high", "dist_52w_low",
    "spy_ret_5d", "spy_ret_21d", "spy_vol_21d", "spy_above_200d",
    "rel_strength_21d", "rel_strength_63d",
]


# ----------------------------- Utilities -----------------------------

def clean_ticker(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def flatten_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # Single ticker downloads sometimes return a multiindex.
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns={c: str(c).title() for c in out.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return pd.DataFrame()
    out = out[required].dropna(subset=["Close"])
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.to_datetime(out.index).normalize()
    return out


def cache_key_for_history(tickers: List[str], start: str) -> str:
    payload = "|".join([start] + sorted([clean_ticker(t) for t in tickers]))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def download_history(
    tickers: List[str],
    start: str,
    cache_dir: str = CACHE_DIR,
    refresh_data: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data in chunks. Returns {ticker: dataframe}. Uses disk cache."""
    history: Dict[str, pd.DataFrame] = {}
    tickers = [clean_ticker(t) for t in tickers]

    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"history_{cache_key_for_history(tickers, start)}.pkl"
        if cache_path.exists() and not refresh_data:
            print(f"Loading cached price history: {cache_path}")
            cached = pd.read_pickle(cache_path)
            if isinstance(cached, dict) and cached:
                print(f"Loaded cached usable history for {len(cached)} tickers.")
                return cached
            print("Cached history file was invalid/empty. Re-downloading.")

    print(f"Downloading history for {len(tickers)} tickers from {start}...")

    chunk_size = 100
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"  batch {i//chunk_size + 1}/{math.ceil(len(tickers)/chunk_size)} ({len(chunk)} tickers)")
        try:
            raw = yf.download(
                tickers=" ".join(chunk),
                start=start,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:
            print(f"    download failed: {exc}")
            continue

        if raw is None or raw.empty:
            continue

        if len(chunk) == 1:
            df = flatten_yf_df(raw)
            if not df.empty:
                history[chunk[0]] = df
            continue

        for t in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0):
                    df = flatten_yf_df(raw[t])
                else:
                    continue
                if len(df) >= 260:
                    history[t] = df
            except Exception:
                continue

    print(f"Downloaded usable history for {len(history)} tickers.")
    if cache_path is not None and history:
        try:
            pd.to_pickle(history, cache_path)
            print(f"Saved price history cache: {cache_path}")
        except Exception as exc:
            print(f"Could not save price history cache: {exc}")
    return history


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean() / close


def macd_features(close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    gap = (macd - signal) / (close + 1e-12)
    return hist / (close + 1e-12), gap


def add_spy_features(spy: pd.DataFrame) -> pd.DataFrame:
    close = spy["Close"]
    out = pd.DataFrame(index=spy.index)
    out["spy_ret_5d"] = close.pct_change(5)
    out["spy_ret_21d"] = close.pct_change(21)
    out["spy_vol_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
    out["spy_above_200d"] = (close > close.rolling(200).mean()).astype(float)
    return out


def build_market_proxy_from_history(history: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a robust fallback market proxy if SPY/IVV/VOO fail to download.

    The proxy is an equal-weight index built from available downloaded stock
    daily returns. It starts at 100 and is used only for broad-market regime
    features when ETF data is unavailable.
    """
    returns = []
    for ticker, df in history.items():
        if ticker in {"SPY", "IVV", "VOO"}:
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna().sort_index()
        if len(close) < 260:
            continue
        r = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty:
            continue
        r.name = ticker
        returns.append(r)

    if not returns:
        return pd.DataFrame()

    ret_frame = pd.concat(returns, axis=1).sort_index()
    # Require at least 20 stocks on a date so one tiny set of names does not
    # define the market proxy.
    ew_ret = ret_frame.mean(axis=1, skipna=True)
    counts = ret_frame.count(axis=1)
    ew_ret = ew_ret[counts >= 20].dropna()
    if ew_ret.empty:
        return pd.DataFrame()

    close = 100.0 * (1.0 + ew_ret).cumprod()
    proxy = pd.DataFrame(index=close.index)
    proxy["Open"] = close
    proxy["High"] = close
    proxy["Low"] = close
    proxy["Close"] = close
    proxy["Volume"] = 0
    proxy.index = pd.to_datetime(proxy.index).normalize()
    return proxy


def get_market_history(history: Dict[str, pd.DataFrame], start: str) -> Tuple[pd.DataFrame, str]:
    """Return SPY-like market history for regime features, with fallbacks."""
    for ticker in ["SPY", "IVV", "VOO"]:
        df = history.get(ticker)
        if df is not None and not df.empty and "Close" in df.columns and len(df) >= 260:
            return df, ticker

    # Sometimes the big multi-ticker download fails only for ETFs. Try direct
    # one-by-one downloads before falling back to a stock-universe proxy.
    for ticker in ["SPY", "IVV", "VOO"]:
        try:
            print(f"Trying direct market proxy download: {ticker}...")
            raw = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
            df = flatten_yf_df(raw)
            if not df.empty and len(df) >= 260:
                history[ticker] = df
                return df, ticker
        except Exception as exc:
            print(f"  {ticker} direct download failed: {exc}")

    print("SPY/IVV/VOO unavailable. Building equal-weight market proxy from downloaded stocks.")
    proxy = build_market_proxy_from_history(history)
    if proxy.empty:
        raise RuntimeError(
            "Could not build market-regime proxy from SPY/IVV/VOO or downloaded stock history."
        )
    history["MARKET_PROXY"] = proxy
    return proxy, "MARKET_PROXY"


def build_features_for_ticker(ticker: str, df: pd.DataFrame, spy_features: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    ret = close.pct_change()
    out = pd.DataFrame(index=df.index)
    out["ticker"] = ticker
    out["close"] = close

    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_21d"] = close.pct_change(21)
    out["ret_63d"] = close.pct_change(63)

    out["vol_10d"] = ret.rolling(10).std() * np.sqrt(252)
    out["vol_21d"] = ret.rolling(21).std() * np.sqrt(252)
    out["vol_63d"] = ret.rolling(63).std() * np.sqrt(252)

    out["atr_pct_14d"] = atr_pct(df, 14)
    out["rsi_14d"] = rsi(close, 14)

    hist, gap = macd_features(close)
    out["macd_hist"] = hist
    out["macd_signal_gap"] = gap

    out["volume_ratio_20d"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-12)
    out["dollar_volume_20d"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    out["dist_ma_20d"] = close / ma20 - 1
    out["dist_ma_50d"] = close / ma50 - 1
    out["dist_ma_200d"] = close / ma200 - 1

    high_252 = close.rolling(252).max()
    low_252 = close.rolling(252).min()
    out["dist_52w_high"] = close / high_252 - 1
    out["dist_52w_low"] = close / low_252 - 1

    # Align SPY features to ticker dates.
    out = out.join(spy_features, how="left")
    out["rel_strength_21d"] = out["ret_21d"] - out["spy_ret_21d"]
    out["rel_strength_63d"] = out["ret_63d"] - spy_features["spy_ret_21d"].reindex(out.index).rolling(3).sum()

    # Target: future 5-trading-day return.
    out["forward_return_5d"] = close.shift(-HOLDING_DAYS) / close - 1
    out["target_up_5pct"] = (out["forward_return_5d"] >= TARGET_GAIN).astype(float)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=FEATURE_COLUMNS + ["target_up_5pct", "forward_return_5d"])
    return out


def get_project_universe(max_tickers: int, universe_mode: str = UNIVERSE_MODE) -> List[str]:
    if get_universe is None:
        # Fallback: liquid US large/mid cap sample.
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "XOM", "UNH", "SPY"]
    else:
        print(f"Loading universe mode: {universe_mode}")
        tickers = get_universe(include_midcap=True, mode=universe_mode)
        # Keep the quick metadata prefilter only for S&P mode. In broad mode, the
        # point is to avoid today's S&P membership survivorship bias, so we rely
        # on historical liquidity/cap filters after downloading prices.
        if universe_mode == "sp" and prefilter_tickers is not None:
            try:
                tickers = prefilter_tickers(tickers)
            except Exception as exc:
                print(f"Prefilter skipped: {exc}")
    tickers = [clean_ticker(t) for t in tickers if clean_ticker(t) != "SPY"]
    tickers = list(dict.fromkeys(tickers))[:max_tickers]
    print(f"Universe size before historical filters: {len(tickers)}")
    return tickers


def estimate_current_shares_outstanding(history: Dict[str, pd.DataFrame], cache_dir: str = CACHE_DIR, refresh_data: bool = False) -> pd.DataFrame:
    """
    Practical point-in-time proxy. yfinance does not provide reliable free
    historical shares-outstanding for every ticker. We estimate shares as:

        current market cap / latest downloaded close

    Then each historical row gets proxy_market_cap = historical close *
    estimated shares. This is not perfect, but it is much better than using
    today's cap bucket for every historical date. It also makes the universe
    audit explicit.
    """
    tickers_for_key = sorted([t for t in history.keys() if t not in {"SPY", "IVV", "VOO", "MARKET_PROXY"}])
    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        key = hashlib.md5("|".join(tickers_for_key).encode("utf-8")).hexdigest()[:12]
        cache_path = Path(cache_dir) / f"cap_info_{key}.csv"
        if cache_path.exists() and not refresh_data:
            print(f"Loading cached market-cap info: {cache_path}")
            cap_info = pd.read_csv(cache_path)
            cap_info.to_csv("universe_current_cap_info.csv", index=False)
            return cap_info

    rows = []
    print("Estimating shares outstanding for historical market-cap proxy...")
    for i, (ticker, df) in enumerate(history.items(), 1):
        if ticker == "SPY" or df is None or df.empty:
            continue
        latest_close = float(df["Close"].dropna().iloc[-1]) if "Close" in df else np.nan
        market_cap = np.nan
        try:
            info = yf.Ticker(ticker).fast_info
            market_cap = info.get("market_cap") or info.get("marketCap") or np.nan
        except Exception:
            pass
        shares_est = market_cap / latest_close if pd.notna(market_cap) and latest_close > 0 else np.nan
        rows.append({
            "ticker": ticker,
            "latest_close": latest_close,
            "current_market_cap": market_cap,
            "estimated_shares_outstanding": shares_est,
        })
        if i % 100 == 0:
            print(f"  estimated cap info for {i}/{len(history)}")
    cap_info = pd.DataFrame(rows)
    cap_info.to_csv("universe_current_cap_info.csv", index=False)
    if cache_path is not None and not cap_info.empty:
        try:
            cap_info.to_csv(cache_path, index=False)
            print(f"Saved market-cap info cache: {cache_path}")
        except Exception as exc:
            print(f"Could not save market-cap info cache: {exc}")
    return cap_info


def cap_bucket(cap: float) -> str:
    if pd.isna(cap):
        return "unknown"
    if cap < 300_000_000:
        return "micro"
    if cap < 2_000_000_000:
        return "small"
    if cap < 10_000_000_000:
        return "mid"
    if cap < 200_000_000_000:
        return "large"
    return "mega"


def apply_historical_cap_proxy_filter(
    data: pd.DataFrame,
    history: Dict[str, pd.DataFrame],
    min_market_cap: float,
    max_market_cap: float,
    cache_dir: str = CACHE_DIR,
    refresh_data: bool = False,
) -> pd.DataFrame:
    cap_info = estimate_current_shares_outstanding(history, cache_dir=cache_dir, refresh_data=refresh_data)
    shares = cap_info.set_index("ticker")["estimated_shares_outstanding"].to_dict()
    data = data.copy()
    data["estimated_shares_outstanding"] = data["ticker"].map(shares)
    data["market_cap_proxy"] = data["close"] * data["estimated_shares_outstanding"]
    data["cap_bucket_proxy"] = data["market_cap_proxy"].map(cap_bucket)
    before = len(data)
    before_tickers = data["ticker"].nunique()
    data = data[(data["market_cap_proxy"] >= min_market_cap) & (data["market_cap_proxy"] <= max_market_cap)].copy()
    after = len(data)
    after_tickers = data["ticker"].nunique()
    print(
        f"Historical cap proxy filter: {before:,} rows/{before_tickers} tickers -> "
        f"{after:,} rows/{after_tickers} tickers "
        f"(${min_market_cap/1e9:.1f}B-${max_market_cap/1e9:.1f}B)"
    )
    audit = (
        data.groupby(["ticker", "cap_bucket_proxy"], as_index=False)
        .agg(rows=("date", "count"), min_date=("date", "min"), max_date=("date", "max"),
             median_market_cap_proxy=("market_cap_proxy", "median"),
             min_market_cap_proxy=("market_cap_proxy", "min"),
             max_market_cap_proxy=("market_cap_proxy", "max"))
        .sort_values("median_market_cap_proxy")
    )
    audit.to_csv("trade_universe_cap_proxy_audit.csv", index=False)
    return data


def build_dataset(
    start: str,
    max_tickers: int,
    universe_mode: str = UNIVERSE_MODE,
    cap_mode: str = CAP_MODE,
    min_market_cap: float = MIN_MARKET_CAP,
    max_market_cap: float = MAX_MARKET_CAP,
    cache_dir: str = CACHE_DIR,
    refresh_data: bool = False,
) -> pd.DataFrame:
    tickers = get_project_universe(max_tickers, universe_mode=universe_mode)
    # Add ETF market proxies for regime features. If they fail, we build an
    # equal-weight proxy from the downloaded stock universe instead of crashing.
    market_proxy_tickers = ["SPY", "IVV", "VOO"]
    all_tickers = sorted(set(tickers + market_proxy_tickers))
    history = download_history(all_tickers, start, cache_dir=cache_dir, refresh_data=refresh_data)

    market_history, market_proxy_name = get_market_history(history, start)
    print(f"Using market-regime proxy: {market_proxy_name}")
    spy_features = add_spy_features(market_history)
    frames = []
    print("Building feature dataset...")
    for i, t in enumerate(tickers, 1):
        df = history.get(t)
        if df is None or len(df) < 300:
            continue
        try:
            features = build_features_for_ticker(t, df, spy_features)
            if not features.empty:
                frames.append(features)
        except Exception as exc:
            print(f"  skipped {t}: {exc}")
        if i % 100 == 0:
            print(f"  processed {i}/{len(tickers)}")

    if not frames:
        raise RuntimeError("No usable training rows were created.")

    data = pd.concat(frames).sort_index()
    data.index.name = "date"
    data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])

    if cap_mode == "proxy":
        data = apply_historical_cap_proxy_filter(data, history, min_market_cap, max_market_cap, cache_dir=cache_dir, refresh_data=refresh_data)
        if data.empty:
            raise RuntimeError("Historical cap proxy filter removed all rows. Relax min/max market cap or use --cap-mode none.")
    else:
        data["market_cap_proxy"] = np.nan
        data["cap_bucket_proxy"] = "not_checked"

    print(f"Dataset: {len(data):,} rows | {data['ticker'].nunique()} tickers | "
          f"{data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Base hit rate: {data['target_up_5pct'].mean()*100:.2f}%")
    data.to_csv("probability_training_dataset.csv", index=False)
    return data


def make_model(model_name: str) -> Pipeline:
    if model_name == "logistic":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ])
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=100,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", clf),
        ])

    # Default: sklearn gradient boosting. No external XGBoost dependency.
    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])


def evaluate_predictions(pred: pd.DataFrame) -> dict:
    y = pred["target_up_5pct"].astype(int)
    p = pred["ml_prob_up_5pct"]
    out = {
        "rows": len(pred),
        "tickers": pred["ticker"].nunique(),
        "base_hit_rate": y.mean(),
    }
    try:
        out["auc"] = roc_auc_score(y, p)
    except Exception:
        out["auc"] = np.nan
    try:
        out["brier"] = brier_score_loss(y, p)
    except Exception:
        out["brier"] = np.nan
    try:
        out["log_loss"] = log_loss(y, p)
    except Exception:
        out["log_loss"] = np.nan

    # Ranking edge: top quintile vs bottom quintile by predicted probability.
    tmp = pred.copy()
    tmp["rank_pct"] = tmp.groupby("date")["ml_prob_up_5pct"].rank(pct=True)
    top = tmp[tmp["rank_pct"] >= 0.80]
    bot = tmp[tmp["rank_pct"] <= 0.20]
    out["top_quintile_hit_rate"] = top["target_up_5pct"].mean() if len(top) else np.nan
    out["bottom_quintile_hit_rate"] = bot["target_up_5pct"].mean() if len(bot) else np.nan
    out["hit_rate_edge"] = out["top_quintile_hit_rate"] - out["bottom_quintile_hit_rate"]
    out["top_quintile_avg_5d_return"] = top["forward_return_5d"].mean() if len(top) else np.nan
    out["bottom_quintile_avg_5d_return"] = bot["forward_return_5d"].mean() if len(bot) else np.nan
    out["return_edge"] = out["top_quintile_avg_5d_return"] - out["bottom_quintile_avg_5d_return"]
    return out


def walk_forward_train(data: pd.DataFrame, model_name: str, min_train_years: int) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    data = data.sort_values("date").copy()
    years = sorted(data["date"].dt.year.unique())
    first_test_year = years[0] + min_train_years
    test_years = [y for y in years if y >= first_test_year]
    if not test_years:
        raise RuntimeError("Not enough years for walk-forward validation.")

    all_preds = []
    summaries = []
    last_model = None
    print(f"Walk-forward training: model={model_name}, first test year={first_test_year}")

    for y in test_years:
        train = data[data["date"].dt.year < y]
        test = data[data["date"].dt.year == y]
        if len(train) < 5000 or len(test) < 100:
            continue

        model = make_model(model_name)
        model.fit(train[FEATURE_COLUMNS], train["target_up_5pct"].astype(int))
        probs = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        pred_cols = ["date", "ticker", "close", "target_up_5pct", "forward_return_5d"]
        for extra_col in ["market_cap_proxy", "cap_bucket_proxy"]:
            if extra_col in test.columns:
                pred_cols.append(extra_col)
        pred = test[pred_cols].copy()
        pred["ml_prob_up_5pct"] = probs
        pred["test_year"] = y
        all_preds.append(pred)

        metrics = evaluate_predictions(pred)
        metrics["test_year"] = y
        metrics["train_rows"] = len(train)
        metrics["test_rows"] = len(test)
        summaries.append(metrics)
        last_model = model

        print(f"  {y}: AUC={metrics['auc']:.3f}, top hit={metrics['top_quintile_hit_rate']:.2%}, "
              f"bottom hit={metrics['bottom_quintile_hit_rate']:.2%}, "
              f"return edge={metrics['return_edge']:.2%}")

    if not all_preds:
        raise RuntimeError("No walk-forward predictions created.")

    predictions = pd.concat(all_preds).sort_values(["date", "ticker"])
    overall = evaluate_predictions(predictions)
    overall["test_year"] = "OVERALL"
    overall["train_rows"] = np.nan
    overall["test_rows"] = len(predictions)
    summaries.append(overall)
    summary_df = pd.DataFrame(summaries)

    # Fit final model on all data for latest live scoring.
    final_model = make_model(model_name)
    final_model.fit(data[FEATURE_COLUMNS], data["target_up_5pct"].astype(int))
    return predictions, summary_df, final_model


def feature_importance(model: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Permutation-like importance using correlation with final probabilities as a simple fallback."""
    try:
        clf = model.named_steps["model"]
        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
            return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": vals}).sort_values("importance", ascending=False)
        if hasattr(clf, "coef_"):
            vals = np.abs(clf.coef_[0])
            return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": vals}).sort_values("importance", ascending=False)
    except Exception:
        pass

    # HistGradientBoosting has no built-in importance; use univariate proxy.
    sample = data.dropna(subset=FEATURE_COLUMNS + ["target_up_5pct"]).sample(
        min(50000, len(data)), random_state=RANDOM_STATE
    )
    rows = []
    y = sample["target_up_5pct"].astype(float)
    for col in FEATURE_COLUMNS:
        x = sample[col].astype(float)
        corr = x.corr(y)
        rows.append({"feature": col, "importance": abs(corr) if pd.notna(corr) else 0.0})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def latest_scores(data: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    latest = data.sort_values("date").groupby("ticker", as_index=False).tail(1).copy()
    latest["ml_prob_up_5pct"] = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest = latest.sort_values("ml_prob_up_5pct", ascending=False)
    cols = ["date", "ticker", "close", "ml_prob_up_5pct"]
    for extra_col in ["market_cap_proxy", "cap_bucket_proxy"]:
        if extra_col in latest.columns:
            cols.append(extra_col)
    cols += FEATURE_COLUMNS[:8]
    return latest[cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START_DATE, help="history start date, default 2018-01-01")
    parser.add_argument("--max-tickers", type=int, default=MAX_TICKERS)
    parser.add_argument("--model", choices=["gbdt", "logistic", "rf"], default="gbdt")
    parser.add_argument("--min-train-years", type=int, default=MIN_TRAIN_YEARS)
    parser.add_argument("--universe", choices=["sp", "broad"], default=UNIVERSE_MODE, help="sp=current S&P 500+400; broad=NASDAQ/NYSE listed names")
    parser.add_argument("--cap-mode", choices=["none", "proxy"], default=CAP_MODE, help="proxy applies historical market-cap proxy filter")
    parser.add_argument("--min-market-cap", type=float, default=MIN_MARKET_CAP)
    parser.add_argument("--max-market-cap", type=float, default=MAX_MARKET_CAP)
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="where downloaded price/cap data is cached; use Google Drive path to persist across Colab restarts")
    parser.add_argument("--refresh-data", action="store_true", help="ignore cached data and re-download")
    args = parser.parse_args()

    print("=" * 70)
    print("TRAIN PROBABILITY MODEL")
    print(f"Target: P(stock gains >= {TARGET_GAIN:.0%} in next {HOLDING_DAYS} trading days)")
    print(f"Start: {args.start} | Model: {args.model} | Max tickers: {args.max_tickers}")
    print(f"Universe: {args.universe} | Cap mode: {args.cap_mode} | Cap range: ${args.min_market_cap:,.0f}-${args.max_market_cap:,.0f}")
    print(f"Data cache: {args.cache_dir} | Refresh data: {args.refresh_data}")
    print("=" * 70)

    data = build_dataset(
        args.start,
        args.max_tickers,
        universe_mode=args.universe,
        cap_mode=args.cap_mode,
        min_market_cap=args.min_market_cap,
        max_market_cap=args.max_market_cap,
        cache_dir=args.cache_dir,
        refresh_data=args.refresh_data,
    )
    predictions, summary, model = walk_forward_train(data, args.model, args.min_train_years)
    importance = feature_importance(model, data)
    latest = latest_scores(data, model)

    predictions.to_csv("ml_probability_predictions.csv", index=False)
    summary.to_csv("probability_model_summary.csv", index=False)
    importance.to_csv("probability_model_feature_importance.csv", index=False)
    latest.to_csv("probability_model_latest_scores.csv", index=False)
    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_gain": TARGET_GAIN,
        "holding_days": HOLDING_DAYS,
    }, "probability_model.pkl")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    display_cols = [
        "test_year", "rows", "base_hit_rate", "auc", "brier",
        "top_quintile_hit_rate", "bottom_quintile_hit_rate", "hit_rate_edge",
        "top_quintile_avg_5d_return", "bottom_quintile_avg_5d_return", "return_edge",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTop feature importance:")
    print(importance.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nLatest top ML probabilities:")
    print(latest.head(30)[["date", "ticker", "close", "ml_prob_up_5pct"]].to_string(
        index=False,
        formatters={"ml_prob_up_5pct": lambda x: f"{x:.1%}", "close": lambda x: f"{x:.2f}"},
    ))

    print("\nSaved:")
    print("  probability_training_dataset.csv")
    print("  ml_probability_predictions.csv")
    print("  probability_model_summary.csv")
    print("  probability_model_feature_importance.csv")
    print("  probability_model_latest_scores.csv")
    print("  probability_model.pkl")
    print("\nNext step: compare probability_model_summary.csv against the current heuristic screener edge.")


if __name__ == "__main__":
    main()
