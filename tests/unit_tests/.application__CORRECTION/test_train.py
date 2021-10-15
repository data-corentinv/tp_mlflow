import unittest
from unittest.mock import patch, Mock, MagicMock
from click.testing import CliRunner
from foodcast.application.train import train


class TestTrain(unittest.TestCase):

    @patch('foodcast.application.train.MultiModel')
    @patch('foodcast.application.train.pd.read_csv')
    @patch('foodcast.application.train.get_run')
    @patch('foodcast.application.train.mlflow.pyfunc')
    @patch('foodcast.application.train.mlflow.sklearn')
    @patch('foodcast.application.train.mlflow')
    def test_train(
        self,
        mock_mlflow: MagicMock,
        mock_sklearn: MagicMock,
        mock_pyfunc: MagicMock,
        mock_get_run: MagicMock,
        mock_read_csv: MagicMock,
        mock_multi_model: MagicMock
    ) -> None:
        mock_run = MagicMock()
        mock_model = Mock()
        mock_multi_model.return_value = mock_model
        mock_mlflow.start_run.return_value = mock_run
        mock_get_run.return_value.info.artifact_uri = 'uri'
        runner = CliRunner()
        result = runner.invoke(train, ['--start-week', '1', '--end-week', '5'])
        assert result.exit_code == 0
        mock_mlflow.start_run.assert_called_once()
        mock_run.__enter__.assert_called_once()
        mock_run.__exit__.assert_called_once()
        mock_mlflow.log_params.assert_called()
        mock_get_run.assert_called()
        mock_read_csv.assert_called()
        mock_multi_model.assert_called()
        mock_model.fit.assert_called()
        mock_sklearn.log_model.assert_called()
        mock_pyfunc.log_model.assert_called()
