import os
import click
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import mlflow.tracking
from mlflow.utils import mlflow_tags
from foodcast.domain.multi_model import MultiModel
from foodcast.application.mlflow_utils import get_run
from foodcast.settings import LOGGING_CONFIGURATION_FILE  # type: ignore
import yaml
import logging
import logging.config
with open(LOGGING_CONFIGURATION_FILE, 'r') as f:  # to import
    logging.config.dictConfig(yaml.safe_load(f.read()))  # to import


@click.command(
    help='Train a forecasting model on a sliding window week by week. Predict on NEXT_WEEK.'
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
def train(
    start_week: int,
    end_week: int,
    n_estimators: int,
    n_models: int,
    degree: int,
    lag_in_week: int
) -> None:

    mlflow_client = mlflow.tracking.MlflowClient()
    with mlflow.start_run(run_name='train') as run:
        logging.info(f"Start mlflow run - train - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'train')
        mlflow.log_params(
            {
                'start_week': start_week,
                'end_week': end_week,
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
        ).set_index('order_date')

        model = MultiModel(
            RandomForestRegressor(n_estimators=10, random_state=42),
            n_models=10
        )
        model.fit(x_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=model.single_estimator,
            artifact_path='simple_model',
        )
        mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path='multi_model',
            code_path=[os.path.join('foodcast', 'domain', 'multi_model.py')],
            conda_env={
                'channels': ['defaults', 'conda-forge'],
                'dependencies': [
                    'mlflow=1.8.0',
                    'numpy=1.17.4',
                    'python=3.7.6',
                    'scikit-learn=0.21.3',
                    'cloudpickle==1.3.0'
                ],
                'name': 'multi-model-env'
            }
        )
        logging.info(f'mlflow.pyfunc.log_model:\n{model}')


if __name__ == '__main__': # pragma: no cover
    train()
