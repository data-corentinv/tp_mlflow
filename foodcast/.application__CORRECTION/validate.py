import os
import click
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.pyfunc
import mlflow.tracking
from mlflow.utils import mlflow_tags
from foodcast.domain.forecast import cross_validate, plotly_predictions
from foodcast.domain.multi_model import MultiModel
from foodcast.application.mlflow_utils import get_run, mlflow_log_pandas, mlflow_log_plotly
from foodcast.settings import LOGGING_CONFIGURATION_FILE  # type: ignore
import yaml
import logging
import logging.config
with open(LOGGING_CONFIGURATION_FILE, 'r') as f:  # to import
    logging.config.dictConfig(yaml.safe_load(f.read()))  # to import


@click.command(
    help='Cross-validate a forecasting model on'
         'a sliding window week by week. Predict on NEXT_WEEK.'
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
    '--n-fold',
    type=click.INT,
    default=10,
    help='Number of temporal cross-validation folds.'
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
def validate(
    start_week: int,
    end_week: int,
    n_fold: int,
    n_estimators: int,
    n_models: int,
    degree: int,
    lag_in_week: int
) -> None:

    mlflow_client = mlflow.tracking.MlflowClient()
    with mlflow.start_run(run_name='validate') as run:
        logging.info(f"Start mlflow run - validate - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'validate')
        mlflow.log_params(
            {
                'start_week': start_week,
                'end_week': end_week,
                'n_fold': n_fold,
                'n_estimators': n_estimators,
                'n_models': n_models,
                'degree': degree,
                'lag_in_week': lag_in_week,
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        features_run = get_run(
            mlflow_client,
            entry_point='features',
            parameters={
                'start_week': start_week,
                'end_week': end_week,
                'degree': degree,
                'lag_in_week': lag_in_week
            },
            git_commit=git_commit
        )
        x_train = pd.read_csv(
            os.path.join(
                features_run.info.artifact_uri,
                'training_set',
                'x_train.csv'
            ),
            parse_dates=['order_date']
        ).set_index('order_date')
        y_train = pd.read_csv(
            os.path.join(
                features_run.info.artifact_uri,
                'training_set',
                'y_train.csv'
            ),
            parse_dates=['order_date']
        ).set_index('order_date')['cash_in']

        model = MultiModel(
            RandomForestRegressor(n_estimators=n_estimators, random_state=42),
            n_models=n_models
        )
        maes, preds_train = cross_validate(model, x_train, y_train, n_fold=n_fold)
        fig = plotly_predictions(preds_train, y_train)
        mlflow_log_plotly(fig, 'plots', 'validation.html')
        for i, mae in enumerate(maes):
            mlflow.log_metric('MAE_MIN', mae.min(), step=i)
            mlflow.log_metric('MAE_MAX', mae.max(), step=i)
            for j, result in enumerate(mae):
                mlflow.log_metric('MAE{}'.format(j), result, step=i)
        mlflow_log_pandas(preds_train.reset_index(), 'cross_validation', 'predictions.csv')


if __name__ == '__main__':  # pragma: no cover
    validate()
