import cvxpy as cp
import numpy as np
import datetime
import pandas as pd
from .StockData import StockData
from . import indicators


class MeanVariance:
    """
    The MeanVariance class performs a Mean-Variance Portfolio Optimization. The optimization follows

    .. math::
        \\max_{\\mathbf{w}} \\Bigg( \\sum_{i=1}^{N} w_i \\mathbb{E}[r_i]
        - \\lambda \\sum_{i=1}^{N}\\sum_{j=1}^{N} w_i w_j \\Sigma_{ij} \\Bigg)

    where :math:`\\mathbf{w}` is the weight vector (portfolio allocation weight), :math:`\\mathbb{E}[r_i]` the expected
    return for asset *i*, :math:`\\lambda` the risk aversion parameter and :math:`\\Sigma_{ij}` the *ij*
    element of the covariance matrix (measure of risk). For more details, see the MeanVariance.pdf file.
    `MeanVariance file <../../../../Notes/Notes.pdf>`_

    """
    def __init__(self,
                 alpha: float = 0.0,
                 risk_aversion: float = 10.0,
                 sigma_target: float = None,
                 period: int = 5,
                 max_weight: float = 0.5,
                 horizon: int = 1,
                 add_cash: bool = True):
        """
        Constructor for the MeanVariance object. In the constructor, the parameters are set. To effectively solve
        the Mean-Variance Portfolio optimization, you need to call :func:`pynanz.MeanVariance.solve`.

        :param float alpha: Decay in the exponential weighting. Set alpha=0.0 for simple averaging. Default=0.0.
        :param float risk_aversion: Risk aversion parameter. Cannot be defined if `sigma_target` is not None. Default=10.0.
        :param float sigma_target: Annualized standard deviation target. Cannot be defined if `risk_aversion` is not
            None. Solving for sigma_target requires bisection algorithm (more computationally demanding). Default=None
        :param int period: Number of data points to consider to compute average and covariance. Default=5.
        :param float max_weight: Maximum weighting that an asset can have. Doesn't apply to the risk-free CASH asset (if
            considered, see `add_cash` parameter). Must be in [0; 1]. Default=0.5.
        :param int horizon: Horizon used to compute the return. See :func:`pynanz.indicators.past_return`. Default=1.
        :param bool add_cash: If true, an extra asset named "CASH" is added as a risk-free, zero-return asset.
            Default=True.
        """

        self.horizon = horizon
        """
        Horizon attribute
        
        :type: int
        :default: lol
        """
        self.alpha = alpha
        self.risk_aversion = risk_aversion
        self.sigma_target = sigma_target
        self.period = period
        self.max_weight = max_weight
        self.add_cash = add_cash

        #############################
        ### PREPARE THE PORTFOLIO ###
        #############################

        # Create a dataframe filled with zeros. Note that the length of this DataFrame is smaller
        # than for self.historical_data. This is due to the fact that we cannot use Markowitz portfolio
        # optimization if we don't have enough historical data !
        cols = self.historical_data.columns.to_list()
        cols += ["risk_aversion", "sigma_day", "sigma_year", "return_day", "return_year"]
        self.portfolio = pd.DataFrame(np.NaN,
                                      index=self.historical_data.index.values[period:],
                                      columns=cols)

        # Prepare the weight vector "beta" using the decay parameter alpha
        self.beta = np.exp(- self.alpha * np.flip(np.arange(0, period)))
        self.beta = self.beta / np.sum(self.beta)

        # Placeholder for the mean vector
        self.mean =None

        # Placeholder for the covariance matrix
        self.cov = None

        for i in range(len(self.portfolio)):
            data = self.historical_data.iloc[i:i+period].to_numpy().transpose()

            mean = np.average(data, axis=1, weights=self.beta)
            cov = np.cov(data, aweights=self.beta)

            print(mean)
            print(cov)
            print()

            if i >= 5:
                break

    def _compute_mean_cov(self, index: int):
        """
        Inplace computation of the mean vector and covariance matrix (self.mean and self.cov), from `index` to
        `index` + `self.period` used for Markowitz optimization. The weighting vector `self.beta` is used to
        weight the time data.

        :param  index: Starting index.
        :type   index: int
        """
        data = self.historical_data.iloc[index: index + self.period].to_numpy().transpose()
        self.mean = np.average(data, axis=1, weights=self.beta)
        self.cov = np.cov(data, aweights=self.beta)


    def solve(self,
              x: StockData,
              mode: str = "global",
              start_date: datetime.datetime = None,
              end_date: datetime.datetime = None,
              delta_date: int = 1):
        """

        :param x:
        :param mode:
        :param start_date:
        :param end_date:
        :param delta_date:
        :return:
        """

        # Case insensitive
        mode = mode.lower()

        # Check if `mode` is allowed
        allowed_mode = ["global", "single shot", "efficient frontier"]
        if mode.lower() not in allowed_mode:
            raise ValueError("mode not allowed, must be in", allowed_mode)

        # Check if start_date is correctly defined if "single shot" or "efficient frontier" mode are activated
        if mode == "single shot" or mode == "efficient frontier":
            if start_date is None:
                raise ValueError("If mode is \"single shot\" or \"efficient frontier\", start_date must be set.")

        # Extract the historical data
        self._extract_historical_data(x)

        # Assign start_date and end_date if necessary
        if start_date is None:
            start_date = self.historical_data.index.values[self.period]
        if end_date is None:
            end_date = datetime.date.today()


        for i in range(self.period, len(self.historical_data), delta_date):
            # If we are not yet at start_date, just skip the current iteration
            if self.historical_data.index[i] < start_date:
                continue

            # If we are over end_date, break out of the loop
            if self.historical_data.index[i] > end_date:
                break










        if mode == "single shot":
            if not x.is_trading_day(start_date):
                raise ValueError("start_date is not a trading day.")






        self.summary = {}
        index_list = []
        self.capital = [1e4]

        date_list = []
        for i in range(len(self.portfolio)):
            self._compute_mean_cov(i)
            date_list.append(self.portfolio.index.values[i])

            if self.sigma_target is None:
                summary = self._one_shot_solver()
            else:
                summary = self._bisection_solver()
            for k in summary:
                if not(k in self.summary):
                    self.summary[k] = []
                self.summary[k].append(summary[k])

        self.summary = pd.DataFrame(self.summary, index=date_list)
        """
        self.capital = [1e4]
        for i in range(1, len(self.summary)):
            w = self.summary.iloc[i-1][self.stock_list].to_numpy().flatten()

            date = self.summary.index[i]
            #r_vec = stock.data.loc[date][[("Adj Close Return", v) for v in self.stock_list]].to_numpy().flatten()

            #r = np.dot(w, r_vec)

            #self.capital.append(self.capital[-1] * (1.0 + r))

        self.summary["capital"] = self.capital"""

    def _extract_historical_data(self, x: StockData):
        """
        Extract historical data from a pynanz.StockData object. The extracted data is stored into the attribute
        `self.historical_data` which is a pd.DataFrame having the same indexing as x.data where each column has
        an asset ticker as label and contain the past return over horizon `self.horizon`. If the attribute
        `self.add_cash` is set to True, then the risk-less asset "CASH" (return = 0 for every datapoint) is added.

        :param StockData x: StockData object containing the target data.
        """
        # Create an empty pd.DataFrame
        self.historical_data = pd.DataFrame(index=x.data.index)

        # Loop over assets
        for stock in x.stock_list:
            # The returns could be with a larger horizon than 1. See indicators.past_return for more info.
            self.historical_data[stock] = indicators.past_return(x.data[(stock, "close")], self.horizon)

        # Add risk-less asset CASH if self.add_cash is True.
        if self.add_cash:
            self.historical_data["CASH"] = np.zeros(shape=len(x.data))
        # Clean-up and sort the historical data
        self.historical_data.dropna(axis=0, inplace=True)
        # Sort the columns
        self.historical_data.sort_index(axis=1, inplace=True)
        # Fetch the stock list
        self.stock_list = self.historical_data.columns.values


    def _bisection_solver(self):
            """
            @brief Bisection method to find a sigma_year target.
            """

            self.risk_aversion = 1e2
            r_left = self._one_shot_solver()
            while r_left["sigma_year"] >= self.sigma_target:
                self.risk_aversion *= 10
                r_left = self._one_shot_solver()
                if self.risk_aversion >= 1e10:
                    return r_left

            self.risk_aversion = 0.0
            r_right = self._one_shot_solver()
            if r_right["sigma_year"] <= self.sigma_target:
                return r_right

            for i in range(0, 100):
                self.risk_aversion = 0.5 * (r_left["risk_aversion"] + r_right["risk_aversion"])
                r_center = self._one_shot_solver()

                if r_center["sigma_year"] >= self.sigma_target:
                    r_right = r_center.copy()
                else:
                    r_left = r_center.copy()

                if np.abs(r_left["risk_aversion"] - r_right["risk_aversion"]) < 1e-4:
                    break

            return r_center


    def _one_shot_solver(self):
        # Fetch the number of stocks
        n_stocks = int(len(self.stock_list))

        # Create a vector of upper-boundary constraint.
        max_weight = np.ones(shape=n_stocks) * self.max_weight
        for i in range(n_stocks):
            # CASH is unconstrained, useful for risky period where bailing-out might be good (crisis)
            if self.stock_list[i] == "CASH":
                max_weight[i] = 1.0

        w = cp.Variable(n_stocks)
        ret = self.mean.T @ w
        risk = cp.quad_form(w, self.cov)
        prob = cp.Problem(cp.Maximize(ret - self.risk_aversion * risk), [cp.sum(w) == 1, w >= 0, w <= max_weight])
        prob.solve()

        summary = {"risk_aversion": self.risk_aversion,
                   "sigma_day": np.sqrt(risk.value),
                   "sigma_year": np.sqrt(risk.value) * np.sqrt(256.0 / self.horizon),
                   "return_day": ret.value,
                   "return_year": np.power(1.0 + ret.value, 256.0 / self.horizon) - 1.0}

        for i in range(0, len(self.stock_list)):
            summary[self.stock_list[i]] = w.value[i]

        return summary
