import cvxpy as cp
import numpy as np
import pandas as pd
from .Market import Market
from . import indicators
from multiprocessing import Pool


class MeanVariance:
    """
    Class to perform Mean-Variance analysis of a given set of asset. The cornerstone of this class is the
    Mean-Variance optimization:

    .. math::
        \\max_{\\mathbf{w}} \\Bigg( \\sum_{i=1}^{N} w_i \\mu_i
        - \\gamma \\sum_{i=1}^{N}\\sum_{j=1}^{N} w_i w_j \\Sigma_{ij} \\Bigg)

    where :math:`\\mathbf{w}` is the weight vector (portfolio allocation weight), :math:`\\mu_i = \\mathbb{E}[r_i]`
    the expected return for asset *i*, :math:`\\gamma` the risk aversion parameter and :math:`\\Sigma_{ij}` the *ij*
    element of the covariance matrix (measure of risk). For more details, see the
    `MeanVariance.pdf <../../../Notes/MeanVariance/MeanVariance.pdf>`_ file.

    """
    def __init__(self,
                 data: pd.DataFrame,
                 alpha: float = 0.0,
                 risk_aversion: float = 10.0,
                 sigma_target: float = None,
                 period: int = 5,
                 max_weight: float = 0.5,
                 horizon: int = 1,
                 add_cash: bool = True):
        """
        Constructor for the MeanVariance object in which the parameters are set. To
        Constructor for the MeanVariance object. In the constructor, the parameters are set. To effectively solve
        the Mean-Variance Portfolio optimization, you need to call :func:`pynanz.MeanVariance.solve`.

        :param pd.DataFrame data: pd.DataFrame containing the historical price data. Index must be a date format,
            the columns must be the ticker name.
        :param float alpha: Decay in the exponential weighting of historical data. Set alpha=0.0 for simple averaging.
            Default = 0.0.
        :param float risk_aversion: Risk aversion parameter. Cannot be defined if `sigma_target` is not None.
            Default = 10.0.
        :param float sigma_target: Annualized standard deviation target. Cannot be defined if `risk_aversion` is not
            None. Solving for sigma_target requires bisection algorithm (more computationally demanding).
            Default = None.
        :param int period: Number of data points to consider to compute average and covariance. Default = 5.
        :param float max_weight: Maximum weighting that an asset can have. Doesn't apply to the risk-free CASH asset (if
            considered, see `add_cash` parameter). Must be in [0; 1]. Default=0.5.
        :param int horizon: Horizon used to compute the return. See :func:`pynanz.indicators.past_return`. Default = 1.
        :param bool add_cash: If true, an extra asset named "CASH" is added as a risk-free, zero-return asset.
            Default=True.
        """

        # Store the parameters
        self._alpha = alpha
        self._risk_aversion = risk_aversion
        self._sigma_target = sigma_target
        if self._risk_aversion is not None and self._sigma_target is not None:
            print("risk_aversion and sigma_target are both defined, sigma_target takes precedence")
        self._period = period
        self._max_weight = max_weight
        self._horizon = horizon

        # Internally store the price data, add CASH if necessary and compute returns
        self.data = data.copy()
        if add_cash and not "CASH" in self.data.columns:
            self.data["CASH"] = 1.0
        for c in self.data.columns:
            self.data[c] = indicators.past_return(self.data[c], horizon=self._horizon)
        self.data.dropna(inplace=True)

        # Save tickers, put integer index for easy slicing
        self.tickers = self.data.columns.copy()

        # Portfolio optimization DataFrame placeholder
        self.portfolio = None

        # Efficient frontier np.ndarray placeholder
        self.efficient_frontier = None

        # Compute the normalized beta weights for average and covariance calculation
        self._beta = np.flip(np.exp(-alpha * np.arange(0, self._period)))
        self._beta = self._beta / np.sum(self._beta)

    @staticmethod
    def solver(mu: np.ndarray,
               sigma: np.ndarray,
               gamma: float,
               max_weight: np.ndarray = None):
        """
        Solve the quadratic problem

        .. math::
            \\max_{\\mathbf{w}} \\Bigg( \\sum_{i=1}^{N} w_i \\mu_i
             - \\gamma \\sum_{i=1}^{N}\\sum_{j=1}^{N} w_i w_j \\Sigma_{ij} \\Bigg)

        where :math:`\\mathbf{w}` is the weight vector to be optimized, :math:`\\boldsymbol{\\mu}` the
        expected return vector, :math:`\\boldsymbol{\\Sigma}` the covariance matrix and :math:`\\gamma`
        the risk aversion parameter.

        The function returns:

        * The portfolio weigths :math:`\\mathbf{w}` (np.ndarray)
        * The expected portfolio return :math:`\\mu_p = \\sum_{i=1}^{N} w_i \\mu_i` (float)
        * The expected portfolio risk (squared) :math:`\\sigma_p^2 = \\sum_{i=1}^{N}\\sum_{j=1}^{N} w_i w_j \\Sigma_{ij}` (float)

        :param np.ndarray mu: Array of expected returns of shape (n_assets,) where n_assets is the number
            of assets considered
        :param np.ndarray sigma: Covariance matrix of shape (n_assets, n_assets).
        :param float gamma: Risk aversion parameter.
        :param np.ndarray max_weight: Array of shape (n_assets,) whose i :sup:`th` entry is the maximum allocation for
            asset i. Default is None (internally handled as np.ones(shape=n_assets))
        :return: weights (np.ndarray), portfolio_return (float), portfolio_risk_squared (float)
        """

        if max_weight is None:
            max_weight = np.ones(shape=len(mu))

        w = cp.Variable(len(mu))            # Portfolio weights
        p_return = mu.T @ w                 # Expected portfolio return
        p_risk = cp.quad_form(w, sigma)     # Expected portfolio risk (squared) "w^T . Sigma . w"

        # Optimization problem (quadratic programming)
        prob = cp.Problem(cp.Maximize(p_return - gamma * p_risk), [cp.sum(w) == 1, w >= 0, w <= max_weight])
        prob.solve()

        return w.value, p_return.value, p_risk.value, gamma

    @staticmethod
    def bisection_solver(mu: np.ndarray,
                         sigma: np.ndarray,
                         risk_target: float,
                         max_weight: np.ndarray = None,
                         max_iter: int = 200,
                         rel_tol: float = 0.001):
        """
        Solve the same quadratic problem as :func:`pynanz.MeanVariance.solver` but targeting a specific value for
        the risk :math:`\\sigma_p` using a bisection algorithm. The risk :math:`\\sigma_p` is defined as

        .. math::
            \\sigma_p = \\Bigg( \\sum_{i=1}^{N}\\sum_{j=1}^{N} w_i w_j \\Sigma_{ij} \\Bigg)^{1/2}

        Take care to the rescaling of :math:`\\sigma_p` depending on the frequency at which it was computed !

        :param np.ndarray mu: Array of expected returns of shape (n_assets,) where n_assets is the number
            of assets considered
        :param np.ndarray sigma: Covariance matrix of shape (n_assets, n_assets).
        :param float risk_target: Risk target parameter.
        :param np.ndarray max_weight: Array of shape (n_assets,) whose i :sup:`th` entry is the maximum allocation for
            asset i. Default is None (internally handled as np.ones(shape=n_assets))
        :param int max_iter: Maximum number of iteration for the bisection algorithm. Default is 200.
        :param float rel_tol: Relative tolerance for the risk target . Default is 0.05.
        :return: weights (np.ndarray), portfolio_return (float), portfolio_risk_squared (float) (similarly
            to :func:`pynanz.MeanVariance.solver`).
        """

        # Target is the square of the risk_target parameter
        target = np.square(risk_target)

        # Set the initial value
        gamma_left = 1e4           # High value = LOW risk (left side of a risk vs return graph)
        gamma_right = 0.0          # Low value = HIGH risk (right side of a risk vs return graph)

        solution_left = MeanVariance.solver(mu, sigma, gamma_left, max_weight)
        solution_right = MeanVariance.solver(mu, sigma, gamma_right, max_weight)

        if target >= solution_right[2]:
            return solution_right

        if target <= solution_left[2]:
            return solution_left

        for i in range(max_iter):
            if i == max_iter - 1:
                print("MAX REACHED")
            gamma_center = 0.5 * (gamma_right + gamma_left)
            solution_center = MeanVariance.solver(mu, sigma, gamma_center, max_weight)

            if np.abs((target - solution_center[2])) < rel_tol * target:
                break
            else:
                if target < solution_center[2]:
                    gamma_right = gamma_center
                else:
                    gamma_left = gamma_center

        return solution_center

    def _compute_mu_sigma(self, data: pd.DataFrame):

        returns = data.values.T

        exp_ret = np.average(returns, axis=1, weights=self._beta)
        cov_mat = np.cov(returns, aweights=self._beta)

        return exp_ret, cov_mat

    def optimize(self,
                 start: pd.Timestamp = None,
                 end: pd.Timestamp = None,
                 frequency: int = 1,
                 n_workers: int = None):
        """
        Solve the Mean-Variance optimization problem from `start` to `end` at frequency `frequency`
        """

        if start is None:
            start = self.data.index[0]
        else:
            start = pd.Timestamp(start)

        if end is None:
            end = self.data.index[-1]
        else:
            end = pd.Timestamp(end)

        # Compute the max weight vector, with no constraint on CASH holding (100% pullout possibility)
        max_weight = np.ones(shape=len(self.tickers)) * self._max_weight
        for i in range(len(self.tickers)):
            if self.tickers[i] == "CASH":
                max_weight[i] = 1.0

        param = self._risk_aversion
        if self._sigma_target is not None:
            # sigma_horizon = sigma_year * sqrt(horizon / 252), necessary rescaling for the bisection algorithm
            param = self._sigma_target * np.sqrt(self._horizon / 252)

        _index = []

        def _generator(data):
            for i, df in enumerate(data.rolling(self._period)):
                if (len(df) < self._period) or (i % frequency != 0) and df.index[-1] < start:
                    continue
                elif df.index[-1] > end:
                    break
                else:
                    exp_ret, cov_mat = self._compute_mu_sigma(df)
                    _index.append(df.index[-1])
                    yield exp_ret, cov_mat, param, max_weight

        with Pool(processes=n_workers) as p:
            if self._sigma_target is not None:
                result = p.starmap(MeanVariance.bisection_solver, _generator(self.data))
            else:
                result = p.starmap(MeanVariance.solver, _generator(self.data))


        self.portfolio = pd.DataFrame(index=_index)
        for i in range(len(self.tickers)):
            ticker = self.tickers[i]
            self.portfolio[ticker] = np.array([x[0][i] for x in result])

        self.portfolio[self.tickers] = self.portfolio[self.tickers].clip(lower=0.0, upper=1.0)
        self.portfolio[self.tickers] = self.portfolio[self.tickers].round(3)

        self.portfolio["yearly_return"] = np.array([x[1] for x in result])
        self.portfolio["yearly_risk"] = np.array([x[2] for x in result])
        self.portfolio["risk_aversion"] = np.array([x[3] for x in result])

        # Rescale the return to obtain the yearly one
        self.portfolio["yearly_return"] = np.power(1.0 + self.portfolio["yearly_return"].values, 252.0 / self._horizon)

        # Rescale the risk to obtain the yearly one
        self.portfolio["yearly_risk"] = np.sqrt(self.portfolio["yearly_risk"].values) * np.sqrt(252.0 / self._horizon)



