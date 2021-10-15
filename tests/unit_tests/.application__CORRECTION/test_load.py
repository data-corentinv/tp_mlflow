import unittest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from foodcast.application.load import load


class TestLoad(unittest.TestCase):

    @patch('foodcast.application.load.mlflow_log_pandas')
    @patch('foodcast.application.load.etl')
    @patch('foodcast.application.load.mlflow')
    def test_load(
        self,
        mock_mlflow: MagicMock,
        mock_etl: MagicMock,
        mock_mlflow_log_pandas: MagicMock
    ) -> None:
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        runner = CliRunner()
        result = runner.invoke(load, ['--start-week', '1', '--end-week', '5'])
        assert result.exit_code == 0
        mock_mlflow.start_run.assert_called_once()
        mock_run.__enter__.assert_called_once()
        mock_run.__exit__.assert_called_once()
        mock_mlflow.log_params.assert_called()
        mock_etl.assert_called_once()
        mock_mlflow_log_pandas.assert_called()
