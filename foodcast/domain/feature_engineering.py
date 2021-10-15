import numpy as np
import pandas as pd
from foodcast.domain.decorators import log_return_shape


@log_return_shape
def dummy_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encoding of the weekday.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Should have an 'order_date' column.

    Returns
    -------
    pd.DataFrame
        Input dataframe with additional one-hot encoding of the weekday.
    """
    df['day'] = df['order_date'].dt.weekday
    df = pd.get_dummies(df, columns=['day'], drop_first=True)
    return df


@log_return_shape
def hour_cos_sin(df: pd.DataFrame, degree: int = 1) -> pd.DataFrame:
    """
    Add sines and cosines of the hours (time represented on a circle).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Should have an 'order_date' column.
    degree : int, optional
        Degree of the sines and cosines computed, by default 1.

    Returns
    -------
    pd.DataFrame
        Input dataframe with 2*degree additional columns representing time.
    """
    omega = 2*np.pi*df['order_date'].dt.hour/24
    for i in range(1, degree + 1):
        df['hour_cos_' + str(i)] = np.cos(i*omega)
        df['hour_sin_' + str(i)] = np.sin(i*omega)
    return df


@log_return_shape
def lag_offline(df: pd.DataFrame, lag_in_week: int = 1) -> pd.DataFrame:
    """
    Compute lagged values in an offline manner, i.e. on a full dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Should have 'order_date' and 'cash_in' columns.
    lag_in_week : int, optional
        Number of weeks to lag, by default 1.

    Returns
    -------
    pd.DataFrame
        Input dataframe with an additional column representing the lagged target.
    """
    df = df.set_index('order_date')
    df[f'lag_{lag_in_week}W'] = df['cash_in'].shift(7*lag_in_week, 'D')
    df = df.dropna()
    df = df.reset_index()
    return df


@log_return_shape
def lag_online(df: pd.DataFrame, past: pd.DataFrame, lag_in_week: int = 1) -> pd.DataFrame:
    """
    Compute lagged values in an online manner, i.e. on a new extract without history.
    The recent history thus need to be loaded.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Should have 'order_date' and 'cash_in' columns.
    past : pd.DataFrame
        Data directly in the past of df.
    lag_in_week : int, optional
        Number of weeks to lag, by default 1.

    Returns
    -------
    pd.DataFrame
        Input dataframe with an additional column representing the lagged target.
    """
    df = df.set_index('order_date')
    past = past.set_index('order_date')
    past = past.shift(7*lag_in_week, 'D')
    df[f'lag_{lag_in_week}W'] = past['cash_in']
    df = df.fillna(0)
    df = df.reset_index()
    return df


@log_return_shape
def features_offline(df: pd.DataFrame, degree: int = 1, lag_in_week: int = 1) -> pd.DataFrame:
    """
    Offline feature engineering with enough history to compute lags.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to add features on.
    degree : int, optional
        Degree of the sines and cosines computed, by default 1.
    lag_in_week : int, optional
        Number of weeks to lag, by default 1.

    Returns
    -------
    pd.DataFrame
        Input dataframe with additional features.
    """
    df = dummy_day(df)
    df = hour_cos_sin(df, degree=degree)
    df = lag_offline(df, lag_in_week=lag_in_week)
    return df


@log_return_shape
def features_online(df: pd.DataFrame, past: pd.DataFrame, degree: int = 1, lag_in_week: int = 1) -> pd.DataFrame:
    """
    Online feature engineering on a data slice without enough history to compute lags.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to add features on.
    past : pd.DataFrame
        Data directly in the past of df.
    degree : int, optional
        Degree of the sines and cosines computed, by default 1.
    lag_in_week : int, optional
        Number of weeks to lag, by default 1.

    Returns
    -------
    pd.DataFrame
        Input dataframe with additional features.
    """
    df = dummy_day(df)
    df = hour_cos_sin(df, degree=degree)
    df = lag_online(df, past, lag_in_week=lag_in_week)
    return df
