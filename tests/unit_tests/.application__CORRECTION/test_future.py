import unittest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from foodcast.application.future import future


class TestFuture(unittest.TestCase):

    @patch('foodcast.application.future.mlflow_log_pandas')
    @patch('foodcast.application.future.features_online')
    @patch('foodcast.application.future.span_future')
    @patch('foodcast.application.future.pd.read_csv')
    @patch('foodcast.application.future.get_run')
    @patch('foodcast.application.future.mlflow')
    def test_future(
        self,
        mock_mlflow: MagicMock,
        mock_get_run: MagicMock,
        mock_read_csv: MagicMock,
        mock_span_future: MagicMock,
        mock_features_online: MagicMock,
        mock_mlflow_log_pandas: MagicMock
    ) -> None:
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_get_run.return_value.info.artifact_uri = 'uri'
        runner = CliRunner()
        result = runner.invoke(future, ['--next-week', '10'])
        assert result.exit_code == 0
        mock_mlflow.start_run.assert_called_once()
        mock_run.__enter__.assert_called_once()
        mock_run.__exit__.assert_called_once()
        mock_mlflow.log_params.assert_called()
        mock_get_run.assert_called()
        mock_read_csv.assert_called()
        mock_span_future.assert_called()
        mock_features_online.assert_called()
        mock_mlflow_log_pandas.assert_called()
