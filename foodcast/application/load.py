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
