import numpy as np
import datetime
import pandas as pd
import yfinance as yf
from typing import Union
from . import indicators
import pickle
import copy


class Market:
    """
    `Market` is a class for downloading, storing and handling market data. Data is downloaded from yahoofinance by
    using `yfinance <https://github.com/ranaroussi/yfinance>`_ as backend. The type of data that is downloaded is

    * Daily *Open, High, Low, Close, Volume (OHLCV)* of diverse financial assets (e.g. stocks, ETFs, ...) that is
    stored in the attribute `self.data` of type pd.DataFrame.
    * Metadata associated to the asset (e.g. sector type) that is stored in the attribute `self.metadata` of type
    pd.DataFrame.

    For `self.data`, the columns are multi-indexed, i.e. each column label is a tuple of len=2 whose first entry is the
    ticker symbol (e.g. "AAPL", "GOOG") and second the attribute (e.g. "open", "high"). Diverse financial indicators
    such as exponentially moving averages (EMA) or moving average convergence divergence (MACD)
    can also be computed.


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

        # List of tickers, e.g. ["AAPL", "GOOG", "USFD", "ARKF"]. Access via property
        self.tickers = None

        # pd.Dataframe containing metadata related to equities
        self.equity = None

        # pd.Dataframe containing metadata related to ETFs
        self.etf = None

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
                 start_date: Union[datetime.date, pd.Timestamp] = None,
                 end_date: Union[datetime.date, pd.Timestamp] = None,
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

        # Yahooquery only considers datetime objects
        if isinstance(start_date, pd.Timestamp):
            start_date = datetime.date(year=start_date.year, month=start_date.month, day=start_date.day)
        if isinstance(end_date, pd.Timestamp):
            end_date = datetime.date(year=end_date.year, month=end_date.month, day=end_date.day)

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

        # Only keep "Open, High, Low, Close, Volume" (OHLCV)
        drop_columns = [x for x in self.data.columns if x[1] not in ["open", "high", "low", "close", "volume"]]
        self.data.drop(columns=drop_columns, inplace=True)
        self.data.sort_index(axis=1, inplace=True)

        # Transform the datetime index into pd.Timestamp
        self.data.index = self.data.index.map(pd.Timestamp)

        # Fetch tickers
        self.tickers = np.unique([x for x, y in self.data.columns])

        # Fetch metadata
        self._fetch_metadata()

    def _fetch_metadata(self):
        t = Ticker(list(self.tickers))

        self.equity = []
        self.etf = []

        # Split between EQUITY and ETF
        quote_types = t.quote_type
        for ticker in self.tickers:
            if quote_types[ticker]["quoteType"].lower() == "equity":
                self.equity.append(ticker)
            elif quote_types[ticker]["quoteType"].lower() == "etf":
                self.etf.append(ticker)
            else:
                raise ValueError(f"{ticker} is neither equity nor etf.")

        self.equity = pd.DataFrame(index=self.equity)
        self.etf = pd.DataFrame(index=self.etf)

        # Get the sector/industry for equities
        t = Ticker(list(self.equity.index))
        tmp = t.summary_profile
        self.equity["industry"] = [tmp[x]["industry"].lower() for x in self.equity.index]
        self.equity["sector"] = [tmp[x]["sector"].lower() for x in self.equity.index]

        # Get the sector/industry for equities
        t = Ticker(list(self.etf.index))
        tmp = t.fund_sector_weightings

        for ticker in self.etf.index:
            if len(self.etf.columns) == 0:
                self.etf[tmp[ticker].index] = 0.0
            self.etf.loc[ticker] = tmp[ticker]

        # rename columns
        rename = dict([(x, x.replace("_", " ")) for x in self.etf.columns])
        rename["realestate"] = "real estate"
        self.etf.rename(columns=rename, inplace=True)

    def row(self, loc: pd.Timestamp, target: str = None, drop_level: bool = True):
        if target:
            return self.data.loc[loc].xs(target, level=1, drop_level=drop_level)
        else:
            return self.data.loc[loc]

    @staticmethod
    def load(filename):
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

    def backtester(self,
                        start_date: pd.Timestamp = None,
                        end_date: pd.Timestamp = None,
                        history: int = None):
        # Default values
        if start_date is None:
            start_date = self.oldest_date()
        if end_date is None:
            end_date = self.newest_date()

        # Initialize the date
        date = start_date

        while date < end_date:
            if self.is_trading_day(date):
                # Fetch history if needed
                history_data = None
                if history:
                    idx = self.data.index.get_loc(date)
                    history_data = self.data.iloc[idx-history+1:idx+1].copy()
                yield date, self.row(date).unstack(level=1), history_data
            # Increment the date
            date += pd.Timedelta(days=1)

