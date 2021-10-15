import numpy as np
import pandas as pd
from foodcast.infrastructure.extract import extract
from foodcast.domain.decorators import log_return_shape


@log_return_shape
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw extract of data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to clean.

    Returns
    -------
    pd.DataFrame
        Cleaned data.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df = df.rename(columns={'order_number': 'order_id'})
    df = df.sort_values('order_date')
    df['total_product_price'] = df['quantity']*df['product_price']
    df['cash_in'] = df.groupby('order_id')['total_product_price'].transform(np.sum)
    df = df.drop(
        columns=['item_name', 'quantity', 'product_price',
                 'total_products', 'total_product_price'],
        errors='ignore'
    )
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


@log_return_shape
def merge(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Combine two different data sources into a single, consistent one.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe to combine. Should have an 'order_id' and an 'order_date' column.
    df2 : pd.Dataframe
        Second dataframe to combine. Should have an 'order_id' and an 'order_date' column.

    Returns
    -------
    pd.DataFrame
        Combined dataframe.
    """
    df = pd.concat([df1, df2], sort=True)
    df = df.drop(columns='order_id')
    df = df.sort_values('order_date')
    df = df.reset_index(drop=True)
    return df


@log_return_shape
def resample(df: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
    """
    Resample a time series dataframe at a given rate (hourly rate by default).

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe. Should have a column 'order_date'.
    freq: str, optional (default: '1H')
        Sampling frequency.

    Returns
    -------
    pd.dataframe
        Resampled dataframe, with one point per hour in 'order_date'.
    """
    return df.resample('1H', on='order_date').sum().reset_index()


@log_return_shape
def etl(data_dir: str, start_week: int, end_week: int) -> pd.DataFrame:
    """
    Load a cleaned temporal slice of data.

    Parameters
    ----------
    data_dir : str
        Data directory path.
    start_week : int
        First week number (included).
    end_week : int
        Last week number (included).

    Returns
    -------
    pd.DataFrame
        Cleaned data slice between start_week and end_week.
    """
    df1 = extract(data_dir, start_week, end_week, 'restaurant_1')
    df2 = extract(data_dir, start_week, end_week, 'restaurant_2')
    df1 = clean(df1)
    df2 = clean(df2)
    df = merge(df1, df2)
    df = resample(df)
    return df
