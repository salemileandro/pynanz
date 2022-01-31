import numpy as np
import datetime
import pandas as pd
import yfinance as yf
from typing import Union, List
from . import indicators
import pickle
import copy
from contextlib import redirect_stdout
import io


class Market:
    """
    `Market` is a class for downloading, storing and handling market data. Data is downloaded from yahoofinance by
    using `yfinance <https://github.com/ranaroussi/yfinance>`_ as backend. Two types of data are downloaded

    * Daily *Open, High, Low, Close, Volume (OHLCV)* of diverse financial assets (e.g. stocks, ETFs, ...) that is
      stored in the attribute `self.data` of type pd.DataFrame.
    * Metadata associated to the asset (e.g. sector type) that is stored in the attribute `self.metadata` of type
      pd.DataFrame.


    For `self.data`, the columns are multi-indexed, i.e. each column label is a tuple of len=2 whose first entry is the
    ticker symbol (e.g. "AAPL", "GOOG") and second the attribute (e.g. "open", "high"). Diverse financial indicators
    such as exponentially moving averages (EMA) or moving average convergence divergence (MACD) can also be computed.

    **Note that:**


    * Only daily data can be retrieved.
    * Ticker symbols are fully capitalized, e.g. "AAPL" (not "aapl").
    * Attribute are lower case, e.g "close" (not "Close"/"CLOSE").


    **List of attributes:**

    :cvar pd.DataFrame self.data: DataFrame containing the data.
    :cvar list self.tickers: List of ticker symbols considered in the self.data pd.DataFrame,
        e.g. ["AAPL", "GOOG", "USFD", "ARKF"]. Symbols are fully capitalized.
    :cvar list self.attributes: List of attributes available, e.g. ["close", "open", ...].
    """
    def __init__(self):
        """
        Constructor for the StockData class. The object needs to be initialized using either

        * :func:`pynanz.Market.download`
        * :func:`pynanz.Market.load` (if a call to :func:`pynanz.Market.save` was done before).
        """

        # pd.DataFrame containing the data.
        self.data = None

        # List of tickers, e.g. ["AAPL", "GOOG", "USFD", "ARKF"].
        self._tickers = None

        # pd.Dataframe containing metadata related to equities
        self.equity = None

        # pd.Dataframe containing metadata related to ETFs
        self.etf = None

    ##################
    ### Properties ###
    ##################

    @property
    def tickers(self):
        return self._tickers

    @tickers.setter
    def tickers(self, tickers):
        # Safe instantiation with type checking
        if isinstance(tickers, str):
            tickers = [tickers]
        elif isinstance(tickers, list) or isinstance(tickers, np.ndarray):
            if not all([isinstance(i, str) for i in tickers]):
                raise TypeError("tickers must be a list/np.ndarray of str")
        else:
            raise TypeError("tickers must be of type str, list[str] or np.ndarray[str], not", type(tickers))

        self._tickers = np.array(tickers)

    @property
    def attributes(self):
        if self.data is None:
            return None
        else:
            return np.unique([y for x, y in self.data.columns])

    @attributes.setter
    def attributes(self, attributes):
        raise PermissionError("self.attributes is read-only.")

    def download(self,
                 tickers: Union[str, List[str]],
                 start: Union[str, pd.Timestamp] = None,
                 end: Union[str, pd.Timestamp] = None,
                 threshold_date: Union[str, pd.Timestamp] = None,
                 remove_nan: bool = True):
        """
        Download data from yahoo finance using `yfinance <https://github.com/ranaroussi/yfinance>`_ as backend.

        :param str,List[str] tickers: List of str or str that contains the ticker symbols of the
            considered assets, e.g. ["AAPL", "GOOG", "USFD", "ARKF"]. No default.
        :param str,pd.Timestamp start: Start date for fetching the data. If None, the API will fetch data as far in
            the past as it can. Default = None.
        :param str,pd.Timestamp end: End date for fetching the data. If unspecified, it is set to today via
            `pd.Timestamp.ceil(pd.Timestamp.today(), "D")`. Default = None.
        :param str,pd.Timestamp threshold_date: If defined, assets that have the first record of historical data that
            is newer than `threshold_date` are ignored.
        :param bool remove_nan: If True, dates with NaN entries are removed. Default is True.
        """
        # Set the tickers list
        self.tickers = tickers

        # Reset the pd.DataFrame to None
        self.data = None

        if start is not None:
            _start = pd.Timestamp(start) - pd.Timedelta(days=7)

        if end is None:
            end = pd.Timestamp.ceil(pd.Timestamp.today(), "D")
        else:
            end = pd.Timestamp(end)

        ###########################
        ### Download OHLCV data ###
        ###########################
        data = []
        for ticker in self.tickers:
            f = io.StringIO()
            with redirect_stdout(f):
                df = yf.download(tickers=ticker, start=_start, end=end, interval="1d")
            if isinstance(df, pd.DataFrame):
                if len(df) == 0:
                    print("Skipping %s, lack of data" % ticker)
                    continue
                if "Dividends" in df.columns:
                    df.drop(columns=["Dividends"], inplace=True)
                if "Stock Splits" in df.columns:
                    df.drop(columns=["Stock Splits"], inplace=True)
                if threshold_date is not None:
                    if df.index[0] > pd.Timestamp(threshold_date):
                        print("Skipping %s starting at" % ticker, df.index[0])
                        continue

                new_cols = pd.MultiIndex.from_tuples([(ticker, c.lower()) for c in df.columns])
                df.columns = new_cols
                data.append(df)

        self.data = pd.concat(data, axis=1)
        if remove_nan:
            self.data.dropna(axis=0, inplace=True)

        if start is not None:
            self.data = self.data[self.data.index > start]

        self.tickers = self.data.columns.get_level_values(level=0).unique().to_numpy()

    def download_metadata(self):
        """
        Download the metadata associated to the tickers stored is self.tickers. If self.tickers is None, the call will
        fail.
        (this functionality is still in development)
        """
        if self.tickers is None:
            raise AttributeError("self.ticker is None, you should call self.download() or self.load() first.")

        self.equity = pd.DataFrame()
        self.etf = pd.DataFrame()
        for ticker in self.tickers:
            x = yf.Ticker(ticker)
            info = x.get_info()
            if info["quoteType"].lower() == "equity":
                d = {'sector': info['sector'].lower(),
                     'ticker': ticker}
                self.equity = self.equity.append(d, ignore_index=True)
            elif info["quoteType"].lower() == "etf":
                d = {"ticker": ticker}
                for i in info["sectorWeightings"]:
                    for k, v in i.items():
                        d[k] = v
                self.etf = self.etf.append(d, ignore_index=True)

        if len(self.equity) == 0:
            self.equity = None
        else:
            self.equity.set_index("ticker", drop=True, inplace=True)

        if len(self.etf) == 0:
            self.etf = None
        else:
            self.etf.set_index("ticker", drop=True, inplace=True)

    def row(self, loc: pd.Timestamp, target: str = None, drop_level: bool = True):
        if target:
            return self.data.loc[loc].xs(target, level=1, drop_level=drop_level)
        else:
            return self.data.loc[loc]

    @classmethod
    def load(cls, filename):
        """
        Load data from a pickle file. Need :func:`pynanz.StockData.save` to be called in a previous run.

        :param str filename: Path to the pickle file, e.g. `"./market_data.pkl"`. No default.
        :return: A pynanz.Market object instance.
        """
        # Read the pickle form the pd.read_pickle() function
        with open(filename, 'rb') as f:
            x = pickle.load(f)
        return x

    def save(self, filename: str):
        """
        Save data to a pickle file (pd.DataFrame.to_pickle call). Retrieve data using :func:`pynanz.StockData.load`.

        :param str filename: Path to the pickle file, e.g. `"./stock_data.pkl"`
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def future_return(self,
                      horizon: int = 1,
                      target: str = "close",
                      force_compute: bool = False):
        """
        For a given target (default=\"close\"), computes the future return over every (\"ticker_symbol\", \"target\")
        columns of the pd.DataFrame self.data. This function applies :func:`pynanz.indicators.future_return`.

        :param int horizon: Time horizon, a positive integer. Default=1.
        :param str target: Name of the targeted attribute (e.g. "close", "open", ...). Default="close".
        :param bool force_compute: If False and self.data contains already the columns, the computation is not executed.
            Default=True.
        :raises: * ValueError if `horizon` <= 0.
                 * ValueError if `target` is not a valid attribute.
        """
        # Check that horizon is a positive number
        if horizon <= 0:
            raise ValueError(f"horizon={horizon} must be > 0.")

        # Fetch attribute list and check if target is within
        if not(target in self.attributes):
            raise ValueError(f"target={target} is an incorrect key.")

        # Set the name of the new column
        name = "future_return_%d" % horizon
        if target != "close":
            name = target + "_" + name

        # If the attribute has already been computed and we don't force, just exit
        if name in self.attributes and not force_compute:
            return

        # Actual computation
        for ticker in self.tickers:
            self.data[(ticker, name)] = indicators.future_return(self.data[(ticker, target)], horizon)

        # Sorting the columns
        self.data.sort_index(axis=1, inplace=True)

    def past_return(self,
                    horizon: int = 1,
                    target: str = "close",
                    force_compute: bool = False):
        """
        For a given target (default=\"close\"), computes the past return over every (\"ticker_symbol\", \"target\")
        columns of the pd.DataFrame self.data. This function applies :func:`pynanz.indicators.past_return`.

        :param int horizon: Time horizon, a positive integer. Default=1.
        :param str target: Name of the targeted attribute (e.g. "close", "open", ...). Default="close".
        :param bool force_compute: If False and self.data contains already the columns, the computation is not executed.
            Default=True.
        :raises: * ValueError if `horizon` <= 0.
                 * ValueError if `target` is not a valid attribute.
        """
        # Check that horizon is a positive number
        if horizon <= 0:
            raise ValueError(f"horizon={horizon} must be > 0.")

        # Fetch attribute list and check if target is within
        if not(target in self.attributes):
            raise ValueError(f"{target} is an incorrect key.")

        # Set the name of the new column
        name = "past_return_%d" % horizon
        if target != "close":
            name = target + "_" + name

        # If the attribute has already been computed and we don't force, just exit
        if name in self.attributes and not force_compute:
            return

        # Actual computation
        for ticker in self.tickers:
            self.data[(ticker, name)] = indicators.past_return(self.data[(ticker, target)], horizon)

        # Sorting the columns
        self.data.sort_index(axis=1, inplace=True)

    def ema(self,
            span: int = 10,
            target: str = "close",
            force_compute: bool = False):
        """
        For a given target (default="close"), computes the Exponentially Moving Average (EMA) for every column
        (\"ticker_symbol\", \"target\") of the pd.DataFrame self.data. This function applies
        :func:`pynanz.indicators.ema`.

        :param int span: Time span, a positive integer. Default=10.
        :param str target: Name of the targeted attribute (e.g. "close", "open", ...). Default="close".
        :param bool force_compute: If False and self.data contains already the columns, the computation is not executed.
            Default=True.
        :raises: * ValueError if `span` <= 0.
                 * ValueError if `target` is not a valid attribute.
        """
        # Check that horizon is a positive number
        if span <= 0:
            raise ValueError(f"span={span} must be > 0.")

        # Fetch attribute list and check if target is within
        if not(target in self.attributes):
            raise ValueError(f"{target} is an incorrect key.")

        # Set the name of the new column
        name = "ema_%d" % span
        if target != "close":
            name = target + "_" + name

        # If the attribute has already been computed and we don't force, just exit
        if name in self.attributes and not force_compute:
            return

        # Actual computation
        for ticker in self.tickers:
            self.data[(ticker, name)] = indicators.ema(self.data[(ticker, target)], span)

        # Sorting the columns
        self.data.sort_index(axis=1, inplace=True)

    def macd(self,
             short_span: int = 12,
             long_span: int = 26,
             signal_span: int = 9,
             target: str = "close",
             force_compute=False):
        """
        For a given target (default="close"), computes the Moving Average Convergence Divergence (MACD) indicator over
        every (\"ticker_symbol\", \"target\") columns of the pd.DataFrame self.data. This function applies
        :func:`pynanz.indicators.macd`.

        This functions creates three new colums: f"macd_{`short_span`}_{`long_span`}_{`signal_span`}",
        f"sig_macd_{`short_span`}_{`long_span`}_{`signal_span`}" and
        f"hist_macd_{`short_span`}_{`long_span`}_{`signal_span`}", which are the MACD, signal and histogram
        respectively. If the target is not "close", the target name will be prepended.

        :param int short_span: Span for the fast EMA curve. Default=12.
        :param int long_span:  Span for the slow EMA curve. Default=26.
        :param int signal_span: Span for the signal line. Default=9.
        :param str target: Target column to compute the indicator. Default="close".
        :param bool force_compute: If False and self.data contains already the columns, the computation is not executed.
            Default=True.
        """
        # Set name
        name = "macd_%d_%d_%d" % (short_span, long_span, signal_span)
        name = [name, "sig_" + name, "hist_" + name]
        if target != "close":
            for i in range(len(name)):
                name[i] = target + "_" + name[i]

        # If all the name is already in the list and we don't force, exit.
        c = 0
        for i in range(len(name)):
            if name[i] in self.attributes and not force_compute:
                c += 1
        if c == int(len(name)):
            return

        # Compute the MACD
        for ticker in self.tickers:
            x = self.data[(ticker, target)]
            macd, sig, hist = indicators.macd(x, short_span, long_span, signal_span)
            self.data[(ticker, name[0])] = macd
            self.data[(ticker, name[1])] = sig
            self.data[(ticker, name[2])] = hist

        # Sort the columns
        self.data.sort_index(axis=1, inplace=True)

    def stochastic(self,
                   period: int = 14,
                   target: str = "close",
                   force_compute: bool = False):
        """
        Computes the stochastic indicator over every (\"ticker_symbol\", \"close\") columns of the pd.DataFrame
        self.data. This function applies :func:`pynanz.indicators.stochastic`.

        This functions creates two new colums: f"stochastic_{`period`}" and f"smoothed_stochastic_{`period`}".
        If the target is not "close", the target name will be prepended.

        :param int period: Period defining the stochastic indicator. Default=14.
        :param str target: Target column to compute the indicator. Default="close".
        :param bool force_compute: If False and self.data contains already the columns, the computation is not executed.
            Default=True.
        :raises:    * NotImplemented if target is not \"close\".
        """
        if target != "close":
            raise NotImplemented("target can only be \"close\".")

        name = "stochastic_%d" % period
        name = [name, "smoothed_%s" % name]
        if target != "close":
            for i in range(len(name)):
                name[i] = target + "_" + name

        # If all the name is already in the list and we don't force, exit.
        c = 0
        for i in range(len(name)):
            if name[i] in self.attributes and not force_compute:
                c += 1
        if c == int(len(name)):
            return

        # Compute the indicator
        for ticker in self.tickers:
            x = self.data[(ticker, "close")]
            x_low = self.data[(ticker, "low")]
            x_high = self.data[(ticker, "high")]
            y, y_smooth = indicators.stochastic(x, x_low, x_high, period)
            self.data[(ticker, name[0])] = y
            self.data[(ticker, name[1])] = y_smooth

        # Sort the columns
        self.data.sort_index(axis=1, inplace=True)

    def oldest_date(self):
        """
        :return: Oldest date which is the first index of the the pd.DataFrame.
        """
        return self.data.index[0]

    def newest_date(self):
        """
        :return: Newest date which is the last index of the the pd.DataFrame.
        """
        return self.data.index[-1]

    def is_trading_day(self, date: Union[datetime.date, pd.Timestamp]):
        """
        Check if the date is an actual trading day, i.e. if it is included in the self.data.index.

        :param datetime.date,pd.Timestamp date: Date to check. Can be a datetime.date object or a pd.Timestamp object.
        :return: True if the date has trading date, False otherwise
         
        """
        if isinstance(date, datetime.date):
            date = pd.Timestamp(year=date.year, month=date.month, day=date.day)

        if not isinstance(date, pd.Timestamp):
            raise TypeError(f"type(date)={type(date)} must be of type datetime.date or pd.Timestamp")

        return bool(date in self.data.index.values)

    def get_price_history(self, attribute: str = "close", add_cash: bool = True):
        if self.data is None:
            raise ValueError("self.data has to be initialized first.")

        df = self.data.xs(attribute, level=1, axis=1).copy()

        if add_cash:
            df["CASH"] = 1.0

        return df
