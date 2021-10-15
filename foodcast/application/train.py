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
