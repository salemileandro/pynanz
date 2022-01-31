import numpy as np
import pandas as pd


def ema(x: pd.Series,
        span: int = 10):
    """
    Compute the exponential moving average (ema) of a pd.Series data.

    :param  pd.Series x: Target data to compute the ema.
    :param  int span: Span to compute the data. The decay alpha=2/(span+1).
    :return: The ema of x as a pd.Series.
    :raises: ValueError if span <= 0. TypeError if type(x) != pd.Series.
    """
    # Sanity checks.
    if not isinstance(x, pd.Series):
        raise TypeError("x must be a pd.Series (x is", type(x), ").")
    if span <= 0:
        raise ValueError("span must be > 0.")

    # Compute the ema based on the exponential moving window `ewm` of pandas.
    y = x.ewm(span=span, adjust=True).mean()

    return y


def future_return(x: pd.Series, horizon: int = 5):
    """
    Compute the future return over a fix time *horizon* of the pd.Series *x*. The future return *y* is defined as

    .. math::
        y[t] = \\frac{ x[t+horizon] - x[t]}{x[t]}

    If `x` has units of value, then `y` can be interpreted as an investment return.
    In general, `y` measure the change over time horizon `horizon` of the quantity `x`.

    :param  x: Target data to compute the future return.
    :type   x: pd.Series.
    :param  horizon: Time horizon, a positive integer.
    :type   horizon: int.
    :return: A pd.Series y.
    :raises: ValueError if horizon <= 0.
    """
    # Sanity check.
    if horizon <= 0:
        raise ValueError(f"horizon={horizon} must be >= 0.")

    y = (x.shift(-horizon) - x) / x

    return y


def past_return(x: pd.Series, horizon: int = 1):
    """
    Compute the past return, that is the realized return, over a fix time `horizon` of the pd.Series `x`.
    The past return `y` is defined as

    .. math::
        y[t] = \\frac{ x[t] - x[t-horizon]}{x[t-horizon]}

    If `x` has units of value, then `y` can be interpreted as an investment return.
    In general, `y` measure the change over time horizon `horizon` of the quantity `x`.

    :param  x: Target data to compute the future return.
    :type   x: pd.Series.
    :param  horizon: Time horizon, a positive integer.
    :type   horizon: int.
    :return: A pd.Series y.
    :raises: ValueError if horizon <= 0.
    """
    # Sanity check.
    if horizon <= 0:
        raise ValueError(f"horizon={horizon} must be >= 0.")

    y = (x - x.shift(horizon)) / x.shift(horizon)

    return y


def macd(x: pd.Series,
         short_span: int = 12,
         long_span: int = 26,
         signal_span: int = 9):
    """
    Compute the Moving Average Convergence Divergence (MACD) indicator. The indicator is defined by a fast ema of span
    `short_span` and a slow ema of span `long_span`. The MACD curve is defined by MACD = short_ema - long_ema.
    The signal line is defined as the ema of span `signal_span` of the MACD curve. The MACD histogram is defined as
    the difference between the MACD curve and the signal line. If the histogram has positive value, it suggests a
    bullish market while a negative histogram suggests a bearish market. Take care that the indicator is not perfect
    (false positive/negative) and typically lags in time (detects momentum reversal after they happen).

    :param  x: Target data to compute the macd.
    :type   x: pd.Series
    :param  short_span: Span for the fast EMA curve. Default=12.
    :type   short_span: int
    :param  long_span:  Span for the slow EMA curve. Default=26.
    :type   long_span:  int
    :param  signal_span: Span for the signal line. Default=9.
    :type   signal_span: int
    :return: MACD, signal_line, histogram, a 3D tuple.
    """

    short_ema = ema(x, short_span)
    long_ema = ema(x, long_span)
    macd = short_ema - long_ema
    signal = ema(macd, signal_span)
    histogram = macd - signal

    return macd, signal, histogram


def stochastic(x: pd.Series,
               x_low: pd.Series = None,
               x_high: pd.Series = None,
               period: int = 14):
    """
    Compute the stochastic indicator as well as a smoothed version (ema of span period/3). The stochastic indicator
    is defined as

    .. math::
        y[t] = 100.0 \\frac{x[t] - \\min(x_{low}[t-period+1:])}{\\max(x_{high}[t-period+1:]) - \\min(x_{low}[t-period+1:])}

    If you don't have access to a separate data for the low/high records of your data, the function uses the data
    itself to compute them. For financial data, where high/low values are reached within a trading period, this might
    not be optimal.

    :param  x: Target data to compute the stochastic indicator.
    :type   x: pd.Series
    :param  x_low: Minimum value of target data to compute the stochastic indicator.
    :type   x_low: pd.Series
    :param  x_high: Maximum value of target data to compute the stochastic indicator.
    :type   x_high: pd.Series
    :param  period: Period over which the stochastic indicator is computed. Default=14.
    :type   period: int
    :return: A tuple (stochastic, stochastic_smoothed).
    """

    if x_low is None:
        x_low = x.copy(deep=True)

    if x_high is None:
        x_high = x.copy(deep=True)

    minimum = x_low.rolling(period).min()
    maximum = x_high.rolling(period).max()
    y = 100.0 * (x - minimum) / (maximum - minimum)

    y_smoothed = ema(y, span=int(period/3.0))

    return y, y_smoothed


def test_func(x):
    """This function will try to calculate:

    .. math::
        \sum_{i=1}^{\\infty} x_{i}

    good luck!
    """
    pass
