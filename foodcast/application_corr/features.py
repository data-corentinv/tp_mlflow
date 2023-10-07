import os
import click
import pandas as pd
import mlflow
import mlflow.tracking
from mlflow.utils import mlflow_tags
from foodcast.domain.feature_engineering import features_offline
from foodcast.application.mlflow_utils import get_run, mlflow_log_pandas
from foodcast.settings import LOGGING_CONFIGURATION_FILE  # type: ignore
import yaml
import logging
import logging.config
with open(LOGGING_CONFIGURATION_FILE, 'r') as f:  # to import
    logging.config.dictConfig(yaml.safe_load(f.read()))  # to import


@click.command(
    help='Perform feature engineering offline and append features to a given dataset.'
)
@click.option(
    '--start-week',
    type=click.INT,
    help='Starting week.'
)
@click.option(
    '--end-week',
    type=click.INT,
    help='Ending week.'
)
@click.option(
    '--degree',
    type=click.INT,
    default=1,
    help='Sinusoidal degree of the feature engineering.'
)
@click.option(
    '--lag-in-week',
    type=click.INT,
    default=1,
    help='Number of weeks to look back for labelled information.'
)
def features(start_week: int, end_week: int, degree: int, lag_in_week: int) -> None:

    mlflow_client = mlflow.tracking.MlflowClient()
    with mlflow.start_run(run_name='features') as run:
        logging.info(f"Start mlflow run - features - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'features')
        mlflow.log_params(
            {
                'start_week': start_week,
                'end_week': end_week,
                'degree': degree,
                'lag_in_week': lag_in_week
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        load_run = get_run(
            mlflow_client,
            entry_point='load',
            parameters={
                'start_week': start_week,
                'end_week': end_week
            },
            git_commit=git_commit
        )
        train_path = os.path.join(
            load_run.info.artifact_uri,
            'data_clean',
            'data.csv'
        )

        train = pd.read_csv(train_path, parse_dates=['order_date'])

        train = features_offline(train, degree=degree, lag_in_week=lag_in_week)
        x_train, y_train = train.drop(columns=['cash_in']), train[['order_date', 'cash_in']]
        mlflow_log_pandas(x_train, 'training_set', 'x_train.csv')
        mlflow_log_pandas(y_train, 'training_set', 'y_train.csv')


if __name__ == '__main__': # pragma: no cover
    features()
