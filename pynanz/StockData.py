import numpy as np
import datetime
import pandas as pd
from yahooquery import Ticker
from typing import Union
from . import indicators


class StockData:
    """
    StockData is a class to store *Open, High, Low, Close, Volume* (OHLCV) data of financial assets (e.g. stocks, ETFs).
    The member function :func:`pynanz.StockData.download` can be called to download data from yahoofinance (uses
    `yahooquery <https://yahooquery.dpguthrie.com>`_ as backend).

    The main attribute of a `StockData` object is `self.data`, which is a pandas.DataFrame object. The index of the
    DataFrame is a list of `datetime.date` object which correspond to **trading** days. The columns are muti-indexed:
    they are tuple of size=2 whose first entry is the `ticker symbol` (e.g. "AAPL", "GOOG") and second the attribute
    (e.g. "open", "high").

    Financial indicators such as exponentially moving averages (EMA) or moving average convergence divergence (MACD)
    can also be computed. Such indicators can be used to build investment strategies (see Strategy class).

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

        * :func:`pynanz.StockData.download`
        * :func:`pynanz.StockData.load` (if a call to :func:`pynanz.StockData.save` was done before).
        """

        # pd.DataFrame containing the data.
        self.data = None

        # List of tickers, e.g. ["AAPL", "GOOG", "USFD", "ARKF"]. Access via property
        self.tickers = None

    @property
    def tickers(self):
        return self._tickers

    @tickers.setter
    def tickers(self, tickers):
        # Default None initialization
        if tickers is None:
            self._tickers = None
            return

        # Safe instantiation
        if isinstance(tickers, str):
            x = [tickers]
        if not (isinstance(tickers, list) or isinstance(tickers, np.ndarray)):
            raise TypeError("tickers must be of type str of list[str], not", type(tickers))
        else:
            if not isinstance(tickers[0], str):
                raise TypeError("tickers must be of type str of list[str], not", type(tickers))

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
                 tickers: Union[list, str],
                 start_date: datetime.date = None,
                 end_date: datetime.date = None,
                 tolerance: int = 15,
                 clean_nan_index: bool = True):
        """
        Download data from yahoo finance using `yahooquery <https://yahooquery.dpguthrie.com>`_ as backend.

        :param list[str],str tickers: List of str or str that contains the ticker symbols of the considered assets, e.g.
            ["AAPL", "GOOG", "USFD", "ARKF"]. No default.
        :param datetime.date start_date: Starting date to consider the data. If unspecified, the API will try to fetch
            starting from the oldest available date. Default = None.
        :param datetime.date end_date: End date (inclusive) for the date to be considered. If unspecified, it is set
            to today (`datetime.date.today()`). Default = None.
        :param int tolerance: If `start_date` is specified, and `tolerance >= 0`, then any asset whose oldest available
            data is newer than than `start_date` up to `tolerance` days is discarded. This option is useful when one
            needs to consider data that exist for long-time enough in their model. Setting `tolerance` a negative value
            deactivate this feature. Default = 15.
        :param bool clean_nan_index: If true, indices (rows) of self.data which contains NaN values are cleaned. This
            can be useful since American and European trading days are not always the same, because of varying holidays.
            Default is True (recommended).
        """
        # Set the tickers list
        self.tickers = tickers

        # Reset the pd.DataFrame to None
        self.data = None

        too_new = []  # Hold the ticker symbols of assets which are too recent (if applicable)
        not_found = []  # Hold the ticker symbols of assets which are not found (if applicable)
        for ticker in self.tickers:
            t = Ticker(ticker)
            if start_date is not None:
                if end_date is None:
                    end_date = datetime.date.today()
                df = t.history(start=start_date, end=end_date, interval="1d")
                day_diff = df.index.values[0][1] - start_date
                if day_diff.days > tolerance > 0:
                    too_new.append(ticker)
                    continue
            else:
                df = ticker.history(period="max", interval="1d")
            if not isinstance(df, pd.DataFrame):
                not_found.append([ticker, df])
                continue
            index = pd.MultiIndex.from_product([[ticker], df.columns.values])
            df.columns = index
            df.reset_index(level=0, drop=True, inplace=True)
            if self.data is None:
                self.data = df.copy()
            else:
                self.data = pd.concat([self.data, df], axis=1)

        # Output some messages to the user.
        if too_new:
            print("Some stocks were too new to be considered:", end=" ")
            for x in too_new:
                print(x, end=", ")
            print()
        if not_found:
            print("Some stocks where not found:", end=" ")
            for x in not_found:
                print(x[0], end=", ")
            print()

        # Clean-up the pd.DataFrame
        self.data.sort_index(axis=0, inplace=True)
        if start_date is not None:
            self.data = self.data[self.data.index >= start_date].copy()
        if clean_nan_index:
            self.data.dropna(axis=0, inplace=True)

        self.tickers = np.unique([x for x, y in self.data.columns])

    def load(self, filename):
        """
        Load data from a pickle file. Need :func:`pynanz.StockData.save` to be called in a previous run.

        :param str filename: Path to the pickle file, e.g. `"./stock_data.pkl"`. No default.
        """
        # Read the pickle form the pd.read_pickle() function
        self.data = pd.read_pickle(filename)
        self.tickers = np.unique([x for x, y in self.data.columns])

    def save(self, filename: str):
        """
        Save data to a pickle file (pd.DataFrame.to_pickle call). Retrieve data using :func:`pynanz.StockData.load`.

        :param str filename: Path to the pickle file, e.g. `"./stock_data.pkl"`
        """
        pd.to_pickle(self.data, filename)

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

    def is_trading_day(self,
                       date: datetime.date = None,
                       day: int = None,
                       month: int = None,
                       year: int = None):
        """
        Check if the date is an actual trading day, i.e. if it is included in the self.data.index.

        :param datetime.date,str date: Date to check. Can be a datetime.date object or a str \"DD-MM-YYYY\". If set to
            None, `day`, `month` and `year` **must** be defined.
        :param int day: Day to check. Integer value from 1 to 31.
        :param int month: Month to check. Integer value from 1 to 12.
        :param int year: Year to check. Integer value, e.g. 2012.
        :return: True if the date is a trading day, False otherwise.
        """
        if all(v is None for v in [date, day, month, year]):
            raise ValueError("data or (day, month, year) must be defined")
        elif date is not None:
            if isinstance(date, str):
                words = date.replace("-", " ").replace("/", " ").replace("\t", " ").split(" ")
                words = list(filter(None, words))
                day = int(words[0])
                month = int(words[1])
                year = int(words[2])
            elif isinstance(date, datetime.datetime):
                day = date.day
                month = date.month
                year = date.year
            else:
                raise TypeError("date must be of type datetime.date, datetime.datetime or str", type(date))
        else:
            if any(v is None for v in [day, month, year]):
                raise ValueError("day, month and year must all be defined if date is not defined.")

        return bool(datetime.date(year=year, month=month, day=day) in self.data.index)
