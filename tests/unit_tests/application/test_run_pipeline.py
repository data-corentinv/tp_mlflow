import numpy as np
import unittest
from unittest.mock import patch, Mock, MagicMock
from click.testing import CliRunner
from foodcast.application.run_pipeline import run_pipeline


class TestRunPipeline(unittest.TestCase):

    @patch('foodcast.application.run_pipeline.mlflow_log_pandas')
    @patch('foodcast.application.run_pipeline.mlflow_log_plotly')
    @patch('foodcast.application.run_pipeline.features_online')
    @patch('foodcast.application.run_pipeline.span_future')
    @patch('foodcast.application.run_pipeline.mlflow.pyfunc')
    @patch('foodcast.application.run_pipeline.mlflow.sklearn')
    @patch('foodcast.application.run_pipeline.plotly_predictions')
    @patch('foodcast.application.run_pipeline.cross_validate')
    @patch('foodcast.application.run_pipeline.MultiModel')
    @patch('foodcast.application.run_pipeline.features_offline')
    @patch('foodcast.application.run_pipeline.etl')
    @patch('foodcast.application.run_pipeline.mlflow')
    def test_run_pipeline(
        self,
        mock_mlflow: MagicMock,
        mock_etl: MagicMock,
        mock_features_offline: MagicMock,
        mock_multi_model: MagicMock,
        mock_cross_validate: MagicMock,
        mock_plotly_predictions: MagicMock,
        mock_sklearn: MagicMock,
        mock_pyfunc: MagicMock,
        mock_span_future: MagicMock,
        mock_features_online: MagicMock,
        mock_mlflow_log_plotly: MagicMock,
        mock_mlflow_log_pandas: MagicMock
    ) -> None:
        mock_run = MagicMock()
        mock_model = Mock()
        mock_multi_model.return_value = mock_model
        mock_mlflow.start_run.return_value = mock_run
        mock_cross_validate.return_value = np.array([[1], [2], [3]]), Mock()
        runner = CliRunner()
        result = runner.invoke(run_pipeline, ['--next-week', '6', '--start-week', '1', '--end-week', '5'])
        assert result.exit_code == 0
        mock_mlflow.start_run.assert_called_once()
        mock_run.__enter__.assert_called_once()
        mock_run.__exit__.assert_called_once()
        mock_mlflow.log_params.assert_called()
        mock_etl.assert_called()
        mock_features_offline.assert_called()
        mock_multi_model.assert_called()
        mock_cross_validate.assert_called()
        mock_plotly_predictions.assert_called()
        mock_model.fit.assert_called()
        mock_sklearn.log_model.assert_called()
        mock_pyfunc.log_model.assert_called()
        mock_span_future.assert_called()
        mock_features_online.assert_called()
        mock_model.predict.assert_called()
        mock_mlflow_log_pandas.assert_called()
        mock_mlflow_log_plotly.assert_called()
        mock_mlflow.log_metric.assert_called()
