import click
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.ensemble import RandomForestRegressor
from foodcast.settings import DATA_DIR, LOGGING_CONFIGURATION_FILE  # type: ignore
from foodcast.domain.transform import etl
from foodcast.domain.feature_engineering import features_offline, features_online
from foodcast.application.mlflow_utils import mlflow_log_pandas, mlflow_log_plotly
from foodcast.domain.forecast import cross_validate, plotly_predictions
from foodcast.domain.multi_model import MultiModel
from foodcast.domain.forecast import span_future
import yaml
import logging
import logging.config
with open(LOGGING_CONFIGURATION_FILE, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))


@click.command(
    help='Run the entire pipeline.'
)
@click.option(
    '--next-week',
    type=click.INT,
    help='Next week to predict on.'
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
def run_pipeline(
    next_week: int,
    start_week: int,
    end_week: int,
    n_fold: int,
    n_estimators: int,
    n_models: int,
    degree: int,
    lag_in_week: int
) -> None:

    with mlflow.start_run(run_name='run_pipeline') as run:
        logging.info(f"Start mlflow run {run.data.tags['mlflow.project.entryPoint']} - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'run_pipeline')
        mlflow.log_params(
            {
                'next_week': next_week,
                'start_week': start_week,
                'end_week': end_week,
                'n_fold': n_fold,
                'n_estimators': n_estimators,
                'n_models': n_models,
                'degree': degree,
                'lag_in_week': lag_in_week,
            }
        )

        # Load
        logging.info(f'Load data...')
        data = etl(DATA_DIR, start_week, end_week)
        mlflow_log_pandas(data, 'data_clean', 'data.csv')

        # Features
        logging.info(f'Build offline features...')
        train = features_offline(data, degree=degree, lag_in_week=lag_in_week)
        x_train, y_train = train.drop(columns=['cash_in']), train[['order_date', 'cash_in']]
        mlflow_log_pandas(x_train, 'training_set', 'x_train.csv')
        mlflow_log_pandas(y_train, 'training_set', 'y_train.csv')
        x_train = x_train.set_index('order_date')
        y_train = y_train.set_index('order_date')['cash_in']

        # Validate
        logging.info(f'Validate model...')
        model = MultiModel(
            RandomForestRegressor(n_estimators=n_estimators, random_state=42),
            n_models=10
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

        # Train
        logging.info(f'Train model...')
        model.fit(x_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=model.single_estimator,
            artifact_path='simple_model',
        )
        mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path='multi_model',
            code_path=['foodcast'],
            conda_env={
                'channels': ['defaults', 'conda-forge'],
                'dependencies': [
                    'python=3.7.6',
                    'mlflow=1.8.0',
                    'numpy=1.17.4',
                    'scikit-learn=0.21.3',
                    'cloudpickle=1.3.0'
                ],
                'name': 'multi-model-env'
            }
        )
        logging.info(f'mlflow.pyfunc.log_model:\n{model}')

        # Future
        logging.info(f'Build future...')
        past = etl(DATA_DIR, next_week - lag_in_week, next_week - 1)
        x_pred = span_future(past['order_date'].max())
        x_pred = features_online(x_pred, past, degree=degree, lag_in_week=lag_in_week)
        mlflow_log_pandas(x_pred, 'prediction_set', 'x_pred.csv')
        x_pred = x_pred.set_index('order_date')
        mlflow_log_pandas(x_pred, 'prediction_set', 'x_pred.json')

        # Predict
        logging.info(f'Predict future...')
        y_pred = model.predict(None, x_pred)
        fig = plotly_predictions(y_pred)
        mlflow_log_plotly(fig, 'plots', 'predictions.html')
        mlflow_log_pandas(y_pred.reset_index(), 'predictions', 'y_pred.csv')


if __name__ == '__main__':  # pragma: no cover
    run_pipeline()
