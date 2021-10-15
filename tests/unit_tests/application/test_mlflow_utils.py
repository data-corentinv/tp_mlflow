import unittest
from unittest.mock import patch, MagicMock, Mock
from mlflow.utils import mlflow_tags
from foodcast.application.mlflow_utils import mlflow_log_pandas, mlflow_log_plotly, get_run
from foodcast.application.mlflow_utils import _match_parameters, _find_existing_run


class TestMlflowUtils(unittest.TestCase):

    @patch('foodcast.application.mlflow_utils.mlflow')
    @patch('foodcast.application.mlflow_utils.tempfile.mkdtemp')
    def test_mlflow_log_pandas_1(self, mock_mkdtemp: MagicMock, mock_mlflow: MagicMock) -> None:
        mock_df = Mock()
        mock_mkdtemp.return_value = 'temp'
        mlflow_log_pandas(mock_df, 'local_dir', 'file.csv')
        mock_mkdtemp.assert_called_once()
        mock_df.to_csv.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once()

    @patch('foodcast.application.mlflow_utils.mlflow')
    @patch('foodcast.application.mlflow_utils.tempfile.mkdtemp')
    def test_mlflow_log_pandas_2(self, mock_mkdtemp: MagicMock, mock_mlflow: MagicMock) -> None:
        mock_df = Mock()
        mock_mkdtemp.return_value = 'temp'
        mlflow_log_pandas(mock_df, 'local_dir', 'file.json')
        mock_mkdtemp.assert_called_once()
        mock_df.to_json.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once()

    @patch('foodcast.application.mlflow_utils.mlflow')
    @patch('foodcast.application.mlflow_utils.tempfile.mkdtemp')
    def test_mlflow_log_pandas_3(self, mock_mkdtemp: MagicMock, mock_mlflow: MagicMock) -> None:
        mock_df = Mock()
        mock_mkdtemp.return_value = 'temp'
        with self.assertRaises(ValueError):
            mlflow_log_pandas(mock_df, 'local_dir', 'file.pkl')
            print('HELLOLOOOO')
            mock_mkdtemp.assert_called_once()

    @patch('foodcast.application.mlflow_utils.mlflow')
    @patch('foodcast.application.mlflow_utils.tempfile.mkdtemp')
    @patch('foodcast.application.mlflow_utils.plotly')
    def test_mlflow_log_plotly(
        self,
        mock_plotly: MagicMock,
        mock_mkdtemp: MagicMock,
        mock_mlflow: MagicMock
    ) -> None:
        mock_fig = Mock()
        mock_mkdtemp.return_value = 'temp'
        mlflow_log_plotly(mock_fig, 'local_dir', 'file.html')
        mock_mkdtemp.assert_called_once()
        mock_plotly.offline.plot.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once()

    def test_match_parameters_1(self) -> None:
        mock_run = Mock()
        mock_run.data.params = {'a': '0', 'b': '1'}
        parameters = {'a': 0, 'b': 1}
        assert _match_parameters(mock_run, parameters)

    def test_match_parameters_2(self) -> None:
        mock_run = Mock()
        mock_run.data.params = {'a': '0', 'c': '1'}
        parameters = {'a': 0, 'b': 1}
        assert not _match_parameters(mock_run, parameters)

    @patch('sys.stdout')
    @patch('foodcast.application.mlflow_utils._match_parameters', side_effect=[True, True])
    @patch('foodcast.application.mlflow_utils._get_experiment_id')
    def test_find_existing_run_1(
        self,
        mock_get_experiment_id: MagicMock,
        mock_match_parameters: MagicMock,
        mock_stdout: MagicMock
    ) -> None:
        mock_mlflow_client = Mock()

        mock_run_1 = Mock()
        mock_run_1.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'train',
            mlflow_tags.MLFLOW_GIT_COMMIT: '1234'
        }
        mock_run_info_1 = Mock()
        mock_run_info_1.status = 'FAILED'

        mock_run_2 = Mock()
        mock_run_2.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'train',
            mlflow_tags.MLFLOW_GIT_COMMIT: '1234'
        }
        mock_run_info_2 = Mock()
        mock_run_info_2.status = 'FINISHED'

        mock_mlflow_client.list_run_infos.return_value = [mock_run_info_1, mock_run_info_2]
        mock_mlflow_client.get_run = Mock(side_effect=[mock_run_1, mock_run_2])

        result = _find_existing_run(mock_mlflow_client, 'train', {'a': '0', 'b': '1'}, '1234')

        mock_get_experiment_id.assert_called_once()
        mock_mlflow_client.list_run_infos.assert_called_once()
        assert mock_mlflow_client.get_run.call_count == 2
        assert mock_match_parameters.call_count == 2
        assert result == mock_run_2

    @patch('sys.stdout')
    @patch('foodcast.application.mlflow_utils._match_parameters', side_effect=[False, True])
    @patch('foodcast.application.mlflow_utils._get_experiment_id')
    def test_find_existing_run_2(
        self,
        mock_get_experiment_id: MagicMock,
        mock_match_parameters: MagicMock,
        mock_stdout: MagicMock
    ) -> None:
        mock_mlflow_client = Mock()

        mock_run_1 = Mock()
        mock_run_1.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'train',
            mlflow_tags.MLFLOW_GIT_COMMIT: '1234'
        }
        mock_run_info_1 = Mock()
        mock_run_info_1.status = 'FINISHED'

        mock_run_2 = Mock()
        mock_run_2.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'train',
            mlflow_tags.MLFLOW_GIT_COMMIT: '1234'
        }
        mock_run_info_2 = Mock()
        mock_run_info_2.status = 'FINISHED'

        mock_mlflow_client.list_run_infos.return_value = [mock_run_info_1, mock_run_info_2]
        mock_mlflow_client.get_run = Mock(side_effect=[mock_run_1, mock_run_2])

        result = _find_existing_run(mock_mlflow_client, 'train', {'a': '0', 'b': '1'}, '1234')

        mock_get_experiment_id.assert_called_once()
        mock_mlflow_client.list_run_infos.assert_called_once()
        assert mock_mlflow_client.get_run.call_count == 2
        assert mock_match_parameters.call_count == 2
        assert result == mock_run_2

    @patch('sys.stdout')
    @patch('foodcast.application.mlflow_utils._match_parameters', side_effect=[True, True])
    @patch('foodcast.application.mlflow_utils._get_experiment_id')
    def test_find_existing_run_3(
        self,
        mock_get_experiment_id: MagicMock,
        mock_match_parameters: MagicMock,
        mock_stdout: MagicMock
    ) -> None:
        mock_mlflow_client = Mock()

        mock_run_1 = Mock()
        mock_run_1.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'validate',
            mlflow_tags.MLFLOW_GIT_COMMIT: '1234'
        }
        mock_run_info_1 = Mock()
        mock_run_info_1.status = 'FINISHED'

        mock_run_2 = Mock()
        mock_run_2.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'train',
            mlflow_tags.MLFLOW_GIT_COMMIT: '1234'
        }
        mock_run_info_2 = Mock()
        mock_run_info_2.status = 'FINISHED'

        mock_mlflow_client.list_run_infos.return_value = [mock_run_info_1, mock_run_info_2]
        mock_mlflow_client.get_run = Mock(side_effect=[mock_run_1, mock_run_2])

        result = _find_existing_run(mock_mlflow_client, 'train', {'a': '0', 'b': '1'}, '1234')

        mock_get_experiment_id.assert_called_once()
        mock_mlflow_client.list_run_infos.assert_called_once()
        assert mock_mlflow_client.get_run.call_count == 2
        assert mock_match_parameters.call_count == 1
        assert result == mock_run_2

    @patch('sys.stdout')
    @patch('foodcast.application.mlflow_utils._match_parameters', side_effect=[True, True])
    @patch('foodcast.application.mlflow_utils._get_experiment_id')
    def test_find_existing_run_4(
        self,
        mock_get_experiment_id: MagicMock,
        mock_match_parameters: MagicMock,
        mock_stdout: MagicMock
    ) -> None:
        mock_mlflow_client = Mock()

        mock_run_1 = Mock()
        mock_run_1.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'train',
            mlflow_tags.MLFLOW_GIT_COMMIT: '9999'
        }
        mock_run_info_1 = Mock()
        mock_run_info_1.status = 'FINISHED'

        mock_run_2 = Mock()
        mock_run_2.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'train',
            mlflow_tags.MLFLOW_GIT_COMMIT: '1234'
        }
        mock_run_info_2 = Mock()
        mock_run_info_2.status = 'FINISHED'

        mock_mlflow_client.list_run_infos.return_value = [mock_run_info_1, mock_run_info_2]
        mock_mlflow_client.get_run = Mock(side_effect=[mock_run_1, mock_run_2])

        result = _find_existing_run(mock_mlflow_client, 'train', {'a': '0', 'b': '1'}, '1234')

        mock_get_experiment_id.assert_called_once()
        mock_mlflow_client.list_run_infos.assert_called_once()
        assert mock_mlflow_client.get_run.call_count == 2
        assert mock_match_parameters.call_count == 2
        assert result == mock_run_2

    @patch('sys.stdout')
    @patch('foodcast.application.mlflow_utils._match_parameters', side_effect=[True, False])
    @patch('foodcast.application.mlflow_utils._get_experiment_id')
    def test_find_existing_run_5(
        self,
        mock_get_experiment_id: MagicMock,
        mock_match_parameters: MagicMock,
        mock_stdout: MagicMock
    ) -> None:
        mock_mlflow_client = Mock()

        mock_run_1 = Mock()
        mock_run_1.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'train',
            mlflow_tags.MLFLOW_GIT_COMMIT: '1234'
        }
        mock_run_info_1 = Mock()
        mock_run_info_1.status = 'FAILED'

        mock_run_2 = Mock()
        mock_run_2.data.tags = {
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT: 'train',
            mlflow_tags.MLFLOW_GIT_COMMIT: '1234'
        }
        mock_run_info_2 = Mock()
        mock_run_info_2.status = 'FINISHED'

        mock_mlflow_client.list_run_infos.return_value = [mock_run_info_1, mock_run_info_2]
        mock_mlflow_client.get_run = Mock(side_effect=[mock_run_1, mock_run_2])

        result = _find_existing_run(mock_mlflow_client, 'train', {'a': '0', 'b': '1'}, '1234')

        mock_get_experiment_id.assert_called_once()
        mock_mlflow_client.list_run_infos.assert_called_once()
        assert mock_mlflow_client.get_run.call_count == 2
        assert mock_match_parameters.call_count == 2
        self.assertIsNone(result)

    @patch('foodcast.application.mlflow_utils.mlflow')
    @patch('foodcast.application.mlflow_utils._find_existing_run')
    def test_get_run_1(
        self,
        mock_find_existing_run: MagicMock,
        mock_mlflow: MagicMock
    ) -> None:
        mock_mlflow_client = Mock()
        mock_find_existing_run.return_value = 'existing_run'
        result = get_run(mock_mlflow_client, 'train', {'a': '0', 'b': '1'}, '1234')
        mock_find_existing_run.assert_called_once()
        assert result == 'existing_run'

    @patch('foodcast.application.mlflow_utils.mlflow')
    @patch('foodcast.application.mlflow_utils._find_existing_run')
    def test_get_run_2(
        self,
        mock_find_existing_run: MagicMock,
        mock_mlflow: MagicMock
    ) -> None:
        mock_mlflow_client = Mock()
        mock_find_existing_run.return_value = None
        mock_submitted_run = Mock()
        mock_mlflow.run.return_value = mock_submitted_run
        expected_run = Mock()
        mock_mlflow_client.get_run.return_value = expected_run
        result = get_run(mock_mlflow_client, 'train', {'a': '0', 'b': '1'}, '1234')
        mock_find_existing_run.assert_called_once()
        mock_mlflow.run.assert_called_once()
        assert result == expected_run
