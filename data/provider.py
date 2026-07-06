"""
Data provider abstraction.

The idea: every other module talks to a DataProvider interface, never
directly to yfinance. When you later want to swap in Polygon.io, IEX,
or Unusual Whales for options flow, you implement a new provider class
here and nothing else in the codebase needs to change.
"""

from abc import ABC, abstractmethod
import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Return a DataFrame with columns: Open, High, Low, Close, Volume
        indexed by date, for the given ticker.
        """
        raise NotImplementedError

    @abstractmethod
    def get_history_bulk(self, tickers: list[str], period: str = "1y", interval: str = "1d") -> dict[str, pd.DataFrame]:
        """
        Return {ticker: DataFrame} for many tickers at once.
        Bulk fetching is much faster than looping get_history for large universes.
        """
        raise NotImplementedError


class YFinanceProvider(DataProvider):
    """
    Free data source. Good for prototyping and backtesting.
    Caveats: occasional gaps/rate limiting, adjusted-close handling needs care,
    not suitable for low-latency live trading.
    """

    def __init__(self):
        import yfinance as yf
        self.yf = yf

    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d",
                     start: str = None, end: str = None) -> pd.DataFrame:
        if start or end:
            df = self.yf.Ticker(ticker).history(start=start, end=end, interval=interval, auto_adjust=True)
        else:
            df = self.yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        return df

    def get_history_bulk(self, tickers: list[str], period: str = "1y", interval: str = "1d",
                          start: str = None, end: str = None) -> dict[str, pd.DataFrame]:
        # yfinance's download() is much faster than looping for large universes,
        # but returns a multi-index column DataFrame we need to unpack per ticker.
        kwargs = {"interval": interval, "auto_adjust": True, "group_by": "ticker",
                  "threads": True, "progress": False}
        if start or end:
            kwargs["start"] = start
            kwargs["end"] = end
        else:
            kwargs["period"] = period

        raw = self.yf.download(tickers, **kwargs)

        result = {}
        if len(tickers) == 1:
            # yfinance doesn't multi-index when only one ticker is requested
            result[tickers[0]] = raw
            return result

        for t in tickers:
            try:
                df = raw[t].dropna(how="all")
                if not df.empty:
                    result[t] = df
            except KeyError:
                # ticker failed to download (delisted, typo, rate-limited, etc.)
                continue
        return result


# Placeholder for later. Implement when you're ready to pay for better data.
# class PolygonProvider(DataProvider):
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#     def get_history(self, ticker, period="1y", interval="1d"):
#         raise NotImplementedError("Wire up Polygon.io REST API here")
#     def get_history_bulk(self, tickers, period="1y", interval="1d"):
#         raise NotImplementedError
