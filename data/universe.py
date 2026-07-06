"""
Universe selection: which tickers we even consider.

Pulls current S&P 500 + S&P 400 (mid-cap) constituents from Wikipedia.
Falls back to a small static list if the network call fails, so the
rest of the pipeline can still be smoke-tested.

Note: index constituents change over time. For a real backtest spanning
multiple years, using TODAY's constituent list on PAST dates introduces
survivorship bias (you'd be screening only stocks that "survived" to be
in the index today). This is a known limitation -- flagged here rather
than hidden. A more rigorous backtest would use point-in-time index
membership data, which typically requires a paid data source.
"""

import pandas as pd
import urllib.request

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-script/1.0)"}


def _fetch_html_tables(url: str) -> list:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req) as resp:
        html = resp.read()
    return pd.read_html(html)


_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "V", "UNH",
    "HD", "PG", "MA", "DIS", "BAC", "XOM", "PFE", "KO", "PEP", "COST", "MRK",
]


def get_broad_market_tickers(include_etfs: bool = False) -> list[str]:
    """
    Pulls the full NASDAQ + NYSE/AMEX listed-securities directory from
    NASDAQ's public trader FTP-over-HTTPS files. This is NOT limited to
    S&P 500/400 -- it includes small caps, micro caps, and everything
    else listed on these exchanges (several thousand tickers).

    Source: nasdaqtrader.com publishes these as pipe-delimited text files,
    updated daily, no auth required.

    Note: this is a much noisier universe than the S&P lists -- it
    includes very thinly-traded and low-quality names. The liquidity
    filter (apply_liquidity_filter) becomes much more important when
    using this universe, not optional.
    """
    urls = {
        "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",  # NYSE, AMEX, etc.
    }
    all_tickers = []
    for name, url in urls.items():
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req) as resp:
                text = resp.read().decode("utf-8")
            lines = text.strip().split("\n")
            header = lines[0].split("|")
            symbol_col = "Symbol" if "Symbol" in header else "ACT Symbol"
            sym_idx = header.index(symbol_col)
            etf_idx = header.index("ETF") if "ETF" in header else None
            test_idx = header.index("Test Issue") if "Test Issue" in header else None

            for line in lines[1:-1]:  # last line is a footer ("File Creation Time...")
                fields = line.split("|")
                if len(fields) <= sym_idx:
                    continue
                symbol = fields[sym_idx].strip()
                if not symbol or "$" in symbol or "." in symbol:
                    continue  # skip warrants/units/preferred share notations
                if test_idx is not None and fields[test_idx].strip() == "Y":
                    continue  # skip test issues
                if not include_etfs and etf_idx is not None and fields[etf_idx].strip() == "Y":
                    continue
                all_tickers.append(symbol)
        except Exception as e:
            print(f"[universe] Failed to fetch {name} listing ({e}); skipping.")

    return sorted(set(all_tickers))


def get_sp500_tickers() -> list[str]:
    try:
        tables = _fetch_html_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception as e:
        print(f"[universe] Failed to fetch S&P 500 list ({e}); using fallback list.")
        return _FALLBACK_TICKERS


def get_sp400_tickers() -> list[str]:
    try:
        tables = _fetch_html_tables("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception as e:
        print(f"[universe] Failed to fetch S&P 400 list ({e}); skipping mid-caps.")
        return []


def get_universe(include_midcap: bool = True, mode: str = "sp") -> list[str]:
    """
    mode: "sp" = S&P 500 (+ S&P 400 if include_midcap) -- well-known, liquid,
          heavily-followed names.
          "broad" = full NASDAQ + NYSE/AMEX listed universe -- thousands of
          tickers including small/micro caps, NOT limited to any index.
          Much noisier; liquidity filtering matters a lot more here.
    """
    if mode == "broad":
        return get_broad_market_tickers()

    tickers = get_sp500_tickers()
    if include_midcap:
        tickers += get_sp400_tickers()
    # dedupe, keep order stable
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def get_market_caps(tickers: list[str]) -> dict:
    """
    Fetches market cap per ticker via yfinance. This is a per-ticker call
    (slower than bulk price history), so only run it on a name list
    that's already been narrowed down by a liquidity filter.
    """
    import yfinance as yf
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            mc = info.get("market_cap") or info.get("marketCap")
            if mc:
                caps[t] = mc
        except Exception:
            continue
    return caps


def apply_market_cap_filter(tickers: list[str], min_cap: float = None, max_cap: float = None) -> list[str]:
    """
    Filter tickers by market capitalization.

    Common bands (approximate, definitions vary by source):
      micro cap:  < $300M
      small cap:  $300M - $2B
      mid cap:    $2B - $10B
      large cap:  $10B - $200B
      mega cap:   > $200B

    Example: apply_market_cap_filter(tickers, min_cap=300e6, max_cap=2e9)
    targets small caps specifically.
    """
    caps = get_market_caps(tickers)
    keep = []
    for t, cap in caps.items():
        if min_cap is not None and cap < min_cap:
            continue
        if max_cap is not None and cap > max_cap:
            continue
        keep.append(t)
    return keep


def apply_liquidity_filter(price_data: dict[str, "pd.DataFrame"],
                            min_price: float = 5.0,
                            max_price: float = None,
                            min_avg_dollar_volume: float = 5_000_000,
                            lookback_days: int = 20) -> list[str]:
    """
    Filter a universe down to tradable names: avoid penny stocks and
    illiquid names where your own orders would move the price.

    max_price: optional ceiling, useful for targeting lower-priced /
    smaller-cap names instead of mega-caps. None = no ceiling.
    """
    keep = []
    for ticker, df in price_data.items():
        if df is None or df.empty or len(df) < lookback_days:
            continue
        recent = df.tail(lookback_days)
        avg_price = recent["Close"].mean()
        avg_dollar_volume = (recent["Close"] * recent["Volume"]).mean()
        if avg_price < min_price:
            continue
        if max_price is not None and avg_price > max_price:
            continue
        if avg_dollar_volume >= min_avg_dollar_volume:
            keep.append(ticker)
    return keep
