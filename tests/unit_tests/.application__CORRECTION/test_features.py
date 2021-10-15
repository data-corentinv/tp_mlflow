import unittest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from foodcast.application.features import features


class TestFeatures(unittest.TestCase):

    @patch('foodcast.application.features.mlflow_log_pandas')
    @patch('foodcast.application.features.features_offline')
    @patch('foodcast.application.features.pd.read_csv')
    @patch('foodcast.application.features.get_run')
    @patch('foodcast.application.features.mlflow')
    def test_features(
        self,
        mock_mlflow: MagicMock,
        mock_get_run: MagicMock,
        mock_read_csv: MagicMock,
        mock_features_offline: MagicMock,
        mock_mlflow_log_pandas: MagicMock
    ) -> None:
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_get_run.return_value.info.artifact_uri = 'uri'
        runner = CliRunner()
        result = runner.invoke(features, ['--start-week', '1', '--end-week', '5'])
        assert result.exit_code == 0
        mock_mlflow.start_run.assert_called_once()
        mock_run.__enter__.assert_called_once()
        mock_run.__exit__.assert_called_once()
        mock_mlflow.log_params.assert_called()
        mock_get_run.assert_called()
        mock_read_csv.assert_called()
        mock_features_offline.assert_called()
        mock_mlflow_log_pandas.assert_called()
