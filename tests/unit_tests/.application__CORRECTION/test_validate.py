import numpy as np
import unittest
from unittest.mock import patch, Mock, MagicMock
from click.testing import CliRunner
from foodcast.application.validate import validate


class TestValidate(unittest.TestCase):

    @patch('foodcast.application.validate.mlflow_log_pandas')
    @patch('foodcast.application.validate.mlflow_log_plotly')
    @patch('foodcast.application.validate.plotly_predictions')
    @patch('foodcast.application.validate.cross_validate')
    @patch('foodcast.application.validate.MultiModel')
    @patch('foodcast.application.validate.pd.read_csv')
    @patch('foodcast.application.validate.get_run')
    @patch('foodcast.application.validate.mlflow')
    def test_validate(
        self,
        mock_mlflow: MagicMock,
        mock_get_run: MagicMock,
        mock_read_csv: MagicMock,
        mock_multi_model: MagicMock,
        mock_cross_validate: MagicMock,
        mock_plotly_predictions: MagicMock,
        mock_mlflow_log_plotly: MagicMock,
        mock_mlflow_log_pandas: MagicMock
    ) -> None:
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_get_run.return_value.info.artifact_uri = 'uri'
        mock_cross_validate.return_value = np.array([[1], [2], [3]]), Mock()
        runner = CliRunner()
        result = runner.invoke(validate, ['--start-week', '1', '--end-week', '5'])
        assert result.exit_code == 0
        mock_mlflow.start_run.assert_called_once()
        mock_run.__enter__.assert_called_once()
        mock_run.__exit__.assert_called_once()
        mock_mlflow.log_params.assert_called()
        mock_get_run.assert_called()
        mock_read_csv.assert_called()
        mock_multi_model.assert_called()
        mock_cross_validate.assert_called()
        mock_plotly_predictions.assert_called()
        mock_mlflow_log_plotly.assert_called()
        mock_mlflow.log_metric.assert_called()
        mock_mlflow_log_pandas.assert_called()
