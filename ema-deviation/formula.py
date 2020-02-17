import numpy as np
import pandas as pd


def simple_ma(df, value_col, num_days, **kwargs):
    _df = df.copy()
    _df.loc[:, value_col + f'_{num_days}_sma'] = _df[value_col].rolling(num_days, **kwargs).mean()
    return _df

def exp_ma(df, value_col, num_days, **kwargs):
    _df = df.copy()
    _df.loc[:, value_col + f'_{num_days}_ema'] = _df[value_col].ewm(span=num_days, **kwargs).mean()
    return _df


def price_deviation(df, price_col, ma_col, z_score):
    _df = df.copy()
    _df.loc[:, ma_col + '_std'] = _df.loc[:, price_col]/_df.loc[:, ma_col] - 1
    if z_score:
        _df.loc[:, ma_col + '_std'] = (_df.loc[:, ma_col + '_std'] - _df.loc[:, ma_col + '_std'].mean())/_df.loc[:, ma_col + '_std'].std()
    return _df


def forward_return(df, num_days, price_col, annualize, z_score):
    _df = df.copy()
    _df.loc[:, f'{price_col}_{num_days}_return'] = _df[price_col].pct_change(periods=num_days).shift(-200)
    if annualize:
        _df.loc[:, f'{price_col}_{num_days}_return'] = np.exp(
            252 / num_days * np.log(_df.loc[:, f'{price_col}_{num_days}_return'] + 1)) - 1

    if z_score:
        _df.loc[:, f'{price_col}_{num_days}_return'] = (_df.loc[:, f'{price_col}_{num_days}_return'] - _df.loc[:,
                                                                                                       f'{price_col}_{num_days}_return'].mean()) / _df.loc[
                                                                                                                                                   :,
                                                                                                                                                   f'{price_col}_{num_days}_return'].std()
    return _df