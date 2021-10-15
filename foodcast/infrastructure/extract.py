import os
import pandas as pd
from foodcast.domain.decorators import log_return_shape


@log_return_shape
def extract(data_dir: str, start_week: int, end_week: int, prefix: str) -> pd.DataFrame:
    """
    Extract a temporal slice of data for a given data source.

    Parameters
    ----------
    data_dir : str
        Data directory path.
    start_week : int
        First week number (included).
    end_week : int
        Last week number (included).
    prefix : str
        Data source identification (e.g. 'restaurant_1')

    Returns
    -------
    pd.DataFrame
        Temporal slice of data.
    """
    df = pd.DataFrame()
    for i in range(start_week, end_week + 1):
        file_path = os.path.join(data_dir, 'batchs', f'{prefix}_week_{i}.csv')
        if os.path.isfile(file_path):
            batch = pd.read_csv(file_path)
            df = pd.concat([df, batch], sort=True)
    return df
