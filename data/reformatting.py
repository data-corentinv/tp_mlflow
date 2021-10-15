import os
import pathlib
import pandas as pd


def load_data():
    """
    Load raw data.

    Returns
    -------
    tuple
        Two relevant dataframes.
    """
    df1 = pd.read_csv(
        os.path.join('raw', 'restaurant-1-orders.csv'),
        parse_dates=['Order Date']
    )
    df2 = pd.read_csv(
        os.path.join('raw', 'restaurant-2-orders.csv'),
        parse_dates=['Order Date']
    )
    return df1, df2


def add_week_number(df, time_column, year_min):
    """
    Add a week number id to a data frame.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe to add a column to.
    time_column: string
        Name of the datetime column to use.
    year_min: int
        Starting year to begin the week enumeration.

    Returns
    -------
    df: pandas.DataFrame
        The input dataframe with a new 'week_number' column.
    """
    df['year'] = df[time_column].dt.year
    df['week'] = df[time_column].dt.week
    df['year_number'] = df['year'] - year_min
    df['week_number'] = df['week'] + 52*df['year_number']
    df = df.drop(columns=['year', 'week', 'year_number'])
    return df


def to_batch(df, group_column, prefix):
    """
    Split a dataframe into muliple CSV files.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe to split.
    group_column: string
        Column to group by.
    prefix: string
        Prefix to prepend at the beginning of CSV files.        
    """
    for i, group in df.groupby(group_column):
        group\
            .drop(columns=[group_column])\
            .to_csv(
                os.path.join('batchs', '{}_{:03d}.csv'.format(prefix, i)),
                header=True,
                index=False
            )


if __name__ == '__main__':
    resto1, resto2 = load_data()
    year_min = min(
        resto1['Order Date'].dt.year.min(),
        resto2['Order Date'].dt.year.min()
    )
    resto1 = add_week_number(resto1, 'Order Date', year_min)
    resto2 = add_week_number(resto2, 'Order Date', year_min)
    pathlib.Path('batchs').mkdir(parents=True, exist_ok=True)
    for i, resto in enumerate([resto1, resto2]):
        to_batch(resto, 'week_number', 'restaurant_{}_week'.format(i+1))
    print()
    print('Well Done !')
    print('Now you can check that there is a directory called batchs, containining 380 files')
    print('To count files, simply type : ls -1 batchs | wc -l')
    print()
