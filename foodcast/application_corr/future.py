import os
import click
import pandas as pd
import mlflow
import mlflow.tracking
from mlflow.utils import mlflow_tags
from foodcast.domain.forecast import span_future
from foodcast.domain.feature_engineering import features_online
from foodcast.application.mlflow_utils import get_run, mlflow_log_pandas
from foodcast.settings import LOGGING_CONFIGURATION_FILE  # type: ignore
import yaml
import logging
import logging.config
with open(LOGGING_CONFIGURATION_FILE, 'r') as f:  # to import
    logging.config.dictConfig(yaml.safe_load(f.read()))  # to import


@click.command(
    help='Build next week features to predict on.'
)
@click.option(
    '--next-week',
    type=click.INT,
    help='Next week to predict on.'
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
def future(next_week: int, degree: int, lag_in_week: int) -> None:

    mlflow_client = mlflow.tracking.MlflowClient()
    with mlflow.start_run(run_name='future') as run:
        logging.info(f"Start mlflow run - future - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'future')
        mlflow.log_params(
            {
                'next_week': next_week,
                'degree': degree,
                'lag_in_week': lag_in_week,
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        load_run = get_run(
            mlflow_client,
            entry_point='load',
            parameters={
                'start_week': next_week - lag_in_week,
                'end_week': next_week - 1
            },
            git_commit=git_commit
        )
        past_path = os.path.join(
            load_run.info.artifact_uri,
            'data_clean',
            'data.csv'
        )
        past = pd.read_csv(past_path, parse_dates=['order_date'])
        x_pred = span_future(past['order_date'].max())
        x_pred = features_online(x_pred, past, degree=degree, lag_in_week=lag_in_week)
        mlflow_log_pandas(x_pred, 'prediction_set', 'x_pred.csv')
        mlflow_log_pandas(x_pred.set_index('order_date'), 'prediction_set', 'x_pred.json')


if __name__ == '__main__': # pragma: no cover
    future()
