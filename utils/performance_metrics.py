import pandas as pd
import pytz
import numpy as np
import pyfolio as pf
from scipy.stats import skew, kurtosis
import warnings
import yfinance as yf
import pandas_datareader.data as pdr

def localize_to_utc_if_needed(timestamp):
    if timestamp.tzinfo is None:
        return timestamp.tz_localize('UTC')
    elif timestamp.tzinfo != pytz.UTC:
        return timestamp.tz_convert('UTC')
    return timestamp

def calculate_performance_metrics(returns, bmk_ticker, startdate, enddate):
    returns.index = returns.index.map(localize_to_utc_if_needed)

    # Get benchmark returns
    yf.pdr_override()
    bmk = pdr.get_data_yahoo(bmk_ticker, start=returns.index[0], end="2024-06-30")
    bmk.index = bmk.index.map(localize_to_utc_if_needed)
    bmk_ret = bmk['Adj Close'].pct_change()

    startdate_utc = localize_to_utc_if_needed(pd.Timestamp(startdate))
    enddate_utc = localize_to_utc_if_needed(pd.Timestamp(enddate))

    returns = returns.loc[startdate_utc:enddate_utc]
    bmk_ret = bmk_ret.loc[startdate_utc:enddate_utc]

    returns, bmk_ret = returns.align(bmk_ret, join='inner')

    returns = returns.dropna()
    bmk_ret = bmk_ret.dropna()

    returns_series = pd.Series(returns)
    returns_series.index = pd.to_datetime(returns_series.index)
    returns_series.index = returns_series.index.map(localize_to_utc_if_needed)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    returns_series = returns.squeeze()
    bmk_ret_series = bmk_ret.squeeze()

    metrics = {
        'cumulative_return': (1 + returns_series).cumprod()[-1] - 1,
        'annual_return': ((1 + (1 + returns_series).cumprod()[-1] - 1)**(252/len(returns_series))) - 1,
        'sharpe_ratio': pf.timeseries.sharpe_ratio(returns_series),
        'sortino_ratio': pf.timeseries.sortino_ratio(returns_series),
        'skewness': skew(returns_series),
        'kurtosis': kurtosis(returns_series),
        'tail_ratio': pf.timeseries.tail_ratio(returns_series),
        'daily_var': pf.timeseries.value_at_risk(returns_series),
        'max_drawdown': pf.timeseries.max_drawdown(returns_series),
        'benchmark_cumulative_return': (1 + bmk_ret_series).cumprod()[-1] - 1,
        'benchmark_annual_return': ((1 + (1 + bmk_ret_series).cumprod()[-1] - 1)**(252/len(bmk_ret_series))) - 1,
        'benchmark_sharpe_ratio': pf.timeseries.sharpe_ratio(bmk_ret_series),
        'benchmark_sortino_ratio': pf.timeseries.sortino_ratio(bmk_ret_series),
        'benchmark_skewness': skew(bmk_ret_series),
        'benchmark_kurtosis': kurtosis(bmk_ret_series),
        'benchmark_tail_ratio': pf.timeseries.tail_ratio(bmk_ret_series),
        'benchmark_daily_var': pf.timeseries.value_at_risk(bmk_ret_series),
        'benchmark_max_drawdown': pf.timeseries.max_drawdown(bmk_ret_series)
    }

    return metrics

