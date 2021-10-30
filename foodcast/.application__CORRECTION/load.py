import click
import mlflow
from foodcast.settings import DATA_DIR, LOGGING_CONFIGURATION_FILE  # type: ignore
from foodcast.domain.transform import etl
from foodcast.application.mlflow_utils import mlflow_log_pandas
import yaml
import logging
import logging.config
with open(LOGGING_CONFIGURATION_FILE, 'r') as f:  # to import
    logging.config.dictConfig(yaml.safe_load(f.read()))  # to import


@click.command(
    help='Load data in a window delimited by START_WEEK (included) and END_WEEK (included).'
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
def load(start_week: int, end_week: int) -> None:

    with mlflow.start_run(run_name='load') as run:
        logging.info(f"Start mlflow run - load - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'load')
        mlflow.log_params({'start_week': start_week, 'end_week': end_week})
        data = etl(DATA_DIR, start_week, end_week)
        mlflow_log_pandas(data, 'data_clean', 'data.csv')


if __name__ == '__main__':  # pragma: no cover
    load()
