import os
import pynanz as pn
from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class Manager(ABC):
    def __init__(self,
                 tickers: List[str],
                 starting_capital: float = 10000.0):

        self.portfolio = pn.Portfolio(tickers, starting_capital=starting_capital)

        self.check_stop_loss_at = "low"
        self.check_take_profit_at = "high"

    def always_do_at_opening(self,
                             market: pn.Market,
                             date:pd.Timestamp):
