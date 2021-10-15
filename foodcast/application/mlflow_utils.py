import os
import logging
import tempfile
from typing import Dict, Any
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.tracking.fluent import _get_experiment_id
import pandas as pd
import plotly
import plotly.graph_objects as go
logger = logging.getLogger(__name__)


def mlflow_log_pandas(df: pd.DataFrame, artifact_path: str, file_name: str) -> None:
    """
    Save a pandas data frame into a temporary directory.
    Log the temporary directory within the mlflow current run.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    artifact_path : str
        Artifacts subdirectory name.
    filename : str
        File name.
    """
    tmpdir = tempfile.mkdtemp()
    file_path = os.path.join(tmpdir, file_name)
    suffix = file_name.split('.')[-1]
    if suffix == 'json':
        df.to_json(file_path, index=False, orient='split')
    elif suffix == 'csv':
        df.to_csv(file_path, header=True, index=False)
    else:
        raise ValueError('Extension should be json or csv')
    mlflow.log_artifact(local_path=file_path, artifact_path=artifact_path)
    logger.info(f'mlflow_log_pandas: {file_name}')


def mlflow_log_plotly(fig: go.Figure, artifact_path: str, local_path: str) -> None:
    """
    Save a plotly figure in a temporary directory.
    Log the temporary directory within the mlflow current run.

    Parameters
    ----------
    fig : go.Figure
        Figure to save.
    local_path : str
        File name.
    artifact_path : str
        Artifacts subdirectory name.
    """
    tmpdir = tempfile.mkdtemp()
    file_path = os.path.join(tmpdir, local_path)
    plotly.offline.plot(fig, filename=file_path, auto_open=False)
    mlflow.log_artifact(local_path=file_path, artifact_path=artifact_path)
    logger.info(f'mlflow_log_go_figure: {local_path}')


def _match_parameters(run: mlflow.entities.Run, parameters: Dict[str, Any]) -> bool:
    """
    Return True if the run has parameters identical to expectation.

    Parameters
    ----------
    run : mlflow.entities.Run
        The run which parameters must be assessed.
    parameters : Dict[str, Any]
        Expected parameters.

    Returns
    -------
    Boolean
        True if run has parameters equal to expectation.
    """
    for key, value in parameters.items():
        run_value = run.data.params.get(key)
        if run_value != str(value):
            return False
    return True


def _find_existing_run(
    mlflow_client: mlflow.tracking.MlflowClient,
    entry_point: str,
    parameters: Dict[str, Any],
    git_commit: str
) -> mlflow.entities.Run:
    """
    Find an existing run which is already terminated.

    Parameters
    ----------
    mlflow_client : mlflow.tracking.MlflowClient
        MLflow client able to retrieve runs.
    entry_point : str
        Valid entry point defined in the MLproject file.
    parameters : Dict[str, Any]
        A dictionary of parameters, as defined in the MLproject file.
    git_commit : str
        Git commit to match.

    Returns
    -------
    mlflow.entities.Run
        The existing run entity if found, None otherwise.
    """
    experiment_id = _get_experiment_id()
    all_run_infos = mlflow_client.list_run_infos(experiment_id)
    for run_info in all_run_infos:
        old_run = mlflow_client.get_run(run_info.run_id)
        old_tags = old_run.data.tags
        old_entry_point = old_tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT)
        if old_entry_point != entry_point:
            continue
        if not _match_parameters(old_run, parameters):
            continue
        if run_info.status != 'FINISHED':
            continue
        old_commit = old_tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        if old_commit != git_commit:
            continue
        logger.info('Found an existing run')
        return old_run
    logger.info('No existing run found.')
    return None


def get_run(
    mlflow_client: mlflow.tracking.MlflowClient,
    entry_point: str,
    parameters: Dict[str, Any],
    git_commit: str
) -> mlflow.entities.Run:
    """
    Return an mlflow run defined by an entry point and parameters.

    Parameters
    ----------
    mlflow_client : mlflow.tracking.MlflowClient
        MLflow client able to retrieve runs.
    entry_point : str
        Valid entry point defined in the MLproject file.
    parameters : Dict[str, Any]
        A dictionary of parameters, as defined in the MLproject file.
    git_commit : str
        Git commit to match.

    Returns
    -------
    mlflow.entities.Run
        The run entity created by the run.
    """
    logger.info(f'get_run: {entry_point} - parameters = {parameters}')
    existing_run = _find_existing_run(mlflow_client, entry_point, parameters, git_commit)
    if existing_run:
        return existing_run
    submitted_run = mlflow.run(
        '.',
        entry_point=entry_point,
        parameters=parameters
    )
    return mlflow_client.get_run(submitted_run.run_id)
