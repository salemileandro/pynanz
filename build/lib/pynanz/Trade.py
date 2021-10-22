import os
import pandas as pd
import numpy as np
import pynanz as pn
import datetime


class NotEnoughCash(Exception):
    pass


class Trade:
    def __init__(self,
                 market: pn.Market,
                 ticker: str,
                 date: datetime.date,
                 cash_available: float = None,
                 units_to_buy: float = None,
                 cash_to_invest: float = None,
                 buy_at: str = "open"):

        # Sanity cheks
        if units_to_buy is None and cash_to_invest is None:
            msg = "Both `units_to_buy` and `cash_to_invest` cannot be None."
            raise TypeError(msg)
        if (units_to_buy is not None) and (cash_to_invest is not None):
            msg = f"`units_to_buy`={units_to_buy} and `cash_to_invest`={cash_to_invest} cannot be both defined."
            raise TypeError(msg)

        price = market.at[date, (ticker, buy_at)]
        if cash_to_invest is not None:
            units_to_buy = cash_to_invest / price
        else:
            cash_to_invest = units_to_buy * price

        # Check if we have enough cash to open a trade
        if cash_available is not None:
            if cash_to_invest > cash_available:
                msg = f"Trade {ticker}: trying to invest {cash_to_invest} while only {cash_available} is available."
                raise NotEnoughCash(msg)

        self.ticker = ope


