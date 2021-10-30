import os
import click
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.tracking
from mlflow.utils import mlflow_tags
from foodcast.domain.forecast import plotly_predictions
from foodcast.application.mlflow_utils import get_run, mlflow_log_pandas, mlflow_log_plotly
from foodcast.settings import LOGGING_CONFIGURATION_FILE  # type: ignore
import yaml
import logging
import logging.config
with open(LOGGING_CONFIGURATION_FILE, 'r') as f:  # to import
    logging.config.dictConfig(yaml.safe_load(f.read()))  # to import


@click.command(
    help='Predict with forecasting model on a sliding window week by week. Predict on NEXT_WEEK.'
)
@click.option(
    '--next-week',
    type=click.INT,
    help='Next week to predict.'
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
    '--n-estimators',
    type=click.INT,
    default=10,
    help='Number of trees in random forest.'
)
@click.option(
    '--n-models',
    type=click.INT,
    default=10,
    help='Number of models in multi-model.'
)
@click.option(
    '--degree',
    type=click.INT,
    default=1,
    help='Number of sinusoidal components in feature engineering.'
)
@click.option(
    '--lag-in-week',
    type=click.INT,
    default=1,
    help='Lag to consider in feature engineering, in weeks.'
)
def predict(
    next_week: int,
    start_week: int,
    end_week: int,
    n_estimators: int,
    n_models: int,
    degree: int,
    lag_in_week: int,
) -> None:

    mlflow_client = mlflow.tracking.MlflowClient()
    with mlflow.start_run(run_name='predict') as run:
        logging.info(f"Start mlflow run - predict - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'predict')
        mlflow.log_params(
            {
                'next_week': next_week,
                'start_week': start_week,
                'end_week': end_week,
                'n_estimators': n_estimators,
                'n_models': n_models,
                'degree': degree,
                'lag_in_week': lag_in_week,
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        train_run = get_run(
            mlflow_client,
            entry_point='train',
            parameters={
                'start_week': start_week,
                'end_week': end_week,
                'n_estimators': n_estimators,
                'n_models': n_models,
                'degree': degree,
                'lag_in_week': lag_in_week
            },
            git_commit=git_commit
        )
        model = mlflow.pyfunc.load_model(
            os.path.join(
                train_run.info.artifact_uri,
                'multi_model',
            )
        )

        future_run = get_run(
            mlflow_client,
            entry_point='future',
            parameters={
                'next_week': next_week,
                'degree': degree,
                'lag_in_week': lag_in_week
            },
            git_commit=git_commit
        )
        x_pred = pd.read_csv(
            os.path.join(
                future_run.info.artifact_uri,
                'prediction_set',
                'x_pred.csv'
            ),
            parse_dates=['order_date']
        ).set_index('order_date')

        y_pred = model.predict(x_pred)
        fig = plotly_predictions(y_pred)
        mlflow_log_plotly(fig, 'plots', 'predictions.html')
        mlflow_log_pandas(y_pred.reset_index(), 'predictions', 'y_pred.csv')


if __name__ == '__main__':  # pragma: no cover
    predict()
