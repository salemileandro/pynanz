# Pynanz
Author: Leandro Salemi  
E-mail: salemileandro@gmail.com  
Version: 0.0.0

Python library to fetch, manipulate and analyze financial data. 

Ongoing project.

Some documentation is available at https://pynanz.readthedocs.io/en/latest/index.html

## Functionalities

    * Class `pynanz.Market`
        - Download daily data `pynanz.Market.download()`
        - Compute financial indicators:
            - Future return `pynanz.Market.future_return()`
            - Past (=realized) return `pynanz.Market.past_return()`
            - Exponential Moving average `pynanz.Market.ema()`
            - MACD `pynanz.Market.macd`
            - Stochastic `pynanz.Market.stochastic`

    * Class `pynanz.MeanVariance`
        - Solve the mean-variance optimization `pynanz.MeanVariance.optimize()`


