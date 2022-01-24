import os
import pynanz as pn
import pandas as pd
import numpy as np
import datetime
from typing import Union, List


def id_generator():
    x = 0
    while True:
        yield x
        x += 1


class Portfolio:
    """
    The Portfolio class is a class that contain all relevant information about a portfolio held by a Manager object
    (see :class:`pynanz.BaseManager`)
    """
    def __init__(self,
                 tickers: List[str],
                 starting_capital: float = 10000.0):

        # List of tickers contained in `market`
        self.tickers = tickers

        # List of tickers contained in `market`, with the extra label "CASH"
        self.tickers_with_cash = self.tickers + ["CASH"]

        # Unique trade ID. Gets updated everytime
        self.trade_id = id_generator()

        # Initialize the portfolio DataFrame and set the CASH row
        cols = ["weight", "units", "average_price", "market_price", "value"]
        self.pf = pd.DataFrame(0.0, index=self.tickers_with_cash, columns=cols)
        self.pf.at["CASH", ["weight", "average_price", "market_price"]] = 1.0
        self.pf.at["CASH", ["units", "value"]] = starting_capital

        # self.open_trades is a pd.DataFrame containing information on currently open trades. Trades can be opened
        # as a fraction of a share (popular retail oriented brokers, such as Etoro, provide this possibility)
        cols = ["ticker", "open_date", "units", "open_price", "market_price", "max_price", "min_price",
                "value", "stop_loss", "trailing_stop_loss", "take_profit"]
        self.open_trades = pd.DataFrame(columns=cols,
                                        index=pd.MultiIndex(levels=[[], []], codes=[[], []], dtype=[str, int]))

        # The closed_trades DataFrame contains the same information as the open_trades one, plus some additional info.
        # This pd.DataFrame is useful to look back at the performance, as well as tax considerations (realized_profit)
        cols += ["closed_date", "realized_profit", "drawdown", "holding_time"]
        self.closed_trades = pd.DataFrame(columns=cols,
                                          index=pd.MultiIndex(levels=[[], []], codes=[[], []], dtype=[str, int]))

        self.pf_history = pd.DataFrame(columns=self.tickers_with_cash + ["total"])

    @property
    def generate_id(self):
        return next(self.trade_id)

    @property
    def cash(self):
        """
        :return: Total value of cash available.
        """
        return self.pf.at["CASH", "value"]

    @property
    def total_value(self):
        """
        :return: Total value of the portfolio.
        """
        return self.pf["value"].sum()

    def update(self,
               market_prices: pd.Series,
               date: pd.Timestamp,
               update_portfolio_values: bool = True,
               save_porfolio_history: bool = True,
               update_open_trades: bool = False):

        # Update the portfolio value if necessary
        if update_portfolio_values:
            self.update_portfolio_value(market_prices, date, save_porfolio_history)

        # Update the open trades if necessary
        if update_open_trades:
            self.update_open_trades(market_prices)



    def update_portfolio_value(self, market_prices, date: pd.Timestamp, save_porfolio_history: bool = False):
        # Update the prices
        for ticker in self.tickers:
            self.pf.at[ticker, "market_price"] = market_prices[ticker]
        self.pf["value"] = self.pf["market_price"] * self.pf["units"]

        # Update the weights
        self.pf["weight"] = self.pf["value"] / self.total_value
        # Save history
        if save_porfolio_history:
            self.pf_history.loc[date] = self.pf.at[self.tickers_with_cash, "value"]

    def update_open_trades(self, market_prices: pd.Series):
        # Update the "market_price" entry, using the pd.Series redundancy labelling.
        self.open_trades["market_price"] = market_prices[self.open_trades["ticker"]]
        # Update the value of the trade.
        self.open_trades["value"] = self.open_trades["units"] * self.open_trades["market_price"]
        # Update the maximum price observed during trade.
        self.open_trades["max_price"] = self.open_trades[["market_price", "max_price"]].max(axis=1)
        # Update the minimum price observed during trade.
        self.open_trades["min_price"] = self.open_trades[["market_price", "min_price"]].min(axis=1)


    def open_trade(self,
                   ticker: Union[str, List[str]],
                   market_prices: pd.Series,
                   date: pd.Timestamp,
                   units: float = None,
                   investment: float = None,
                   fee: float = 0.002,
                   stop_loss: float = None,
                   trailing_stop_loss: float = None,
                   take_profit: float = None):

        # If ticker is a list, we loop recursively
        if isinstance(ticker, list):
            for t in ticker:
                if not isinstance(t, str):
                    raise TypeError(f"ticker={ticker} must be a list of str.")
                self.open_trade(t, market_prices, date,
                                units=units, investment=investment, fee=fee, stop_loss=stop_loss,
                                trailing_stop_loss=trailing_stop_loss, take_profit=take_profit)
            return

        if units is None and investment is None:
            raise ValueError(f"Both units={units} and investment={investment} cannot be None.")
        if units is not None and investment is not None:
            raise ValueError(f"Both units={units} and investment={investment} cannot be defined.")

        market_price = market_prices[ticker]
        # When buying, the fee makes the price slightly higher
        buy_price = market_price * (1.0 + fee)

        if investment is not None:
            units = investment / buy_price
        investment = units * buy_price

        if investment < self.cash:
            raise ValueError(f"{date}: Trying to buy {investment} worth of {ticker} but cash available={self.cash}.")

        d = {"ticker": ticker,
             "open_date": date,
             "units": units,
             "open_price": buy_price,
             "market_price": market_price,
             "max_price": market_price,
             "min_price": market_price,
             "value": market_price * units,
             "stop_loss": stop_loss,
             "trailing_stop_loss": trailing_stop_loss,
             "take_profit": take_profit}

        self.open_trades.loc[self.generate_id] = d

    def close_trade(self, trade_id: Union[int, List[int]]):
