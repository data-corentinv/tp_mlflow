import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from foodcast.infrastructure.extract import extract


class TestExtract(unittest.TestCase):

    @patch('os.path.isfile')
    @patch('pandas.read_csv')
    def test_extract_1(self, mock_read_csv: MagicMock, mock_is_file: MagicMock) -> None:
        df1 = pd.DataFrame({'a': range(1, 5, 1)})
        df2 = pd.DataFrame({'a': range(1, 10, 2)})
        df3 = pd.DataFrame({'a': range(1, 20, 4)})
        mock_is_file.return_value = True
        mock_read_csv.side_effect = [df1, df2, df3]
        result = extract('', 4, 6, 'restaurant-1')
        expected = pd.concat([df1, df2, df3], sort=True)
        assert mock_read_csv.call_count == 3
        assert mock_is_file.call_count == 3
        pd.testing.assert_frame_equal(result, expected)

    @patch('os.path.isfile')
    def test_extract_2(self, mock_is_file: MagicMock) -> None:
        mock_is_file.return_value = False
        result = extract('', 4, 6, 'restaurant-1')
        expected = pd.DataFrame()
        assert mock_is_file.call_count == 3
        pd.testing.assert_frame_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
