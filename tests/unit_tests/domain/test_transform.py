import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from foodcast.domain.transform import clean, merge, resample, etl


class TestTransform(unittest.TestCase):

    def test_clean(self) -> None:
        df = pd.DataFrame(
            {
                'Order Number': [1, 1, 2],
                'Order Date': [
                    '2019-01-03 17:32:00',
                    '2019-01-03 17:32:00',
                    '2019-01-01 16:14:00'
                ],
                'Quantity': [5, 10, 2],
                'Product Price': [2, 1, 8],
                'Item Name': ['sushis', 'makis', 'chirachi']
            }
        )
        result = clean(df)
        expected = pd.DataFrame(
            {
                'order_id': [2, 1],
                'order_date': [
                    pd.Timestamp('2019-01-01 16:14:00'),
                    pd.Timestamp('2019-01-03 17:32:00')
                ],
                'cash_in': [16, 20]
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_merge(self) -> None:
        df1 = pd.DataFrame(
            {
                'order_id': [404, 301],
                'order_date': [
                    pd.Timestamp('2019-01-01 16:14:00'),
                    pd.Timestamp('2019-01-03 17:32:00')
                ]
            }
        )
        df2 = pd.DataFrame(
            {
                'order_id': [506, 112],
                'order_date': [
                    pd.Timestamp('2019-01-01 16:05:00'),
                    pd.Timestamp('2019-01-03 17:44:00')
                ]
            }
        )
        result = merge(df1, df2)
        expected = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-01-01 16:05:00'),
                    pd.Timestamp('2019-01-01 16:14:00'),
                    pd.Timestamp('2019-01-03 17:32:00'),
                    pd.Timestamp('2019-01-03 17:44:00')
                ]
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_resample(self) -> None:
        df = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-01-01 16:05:00'),
                    pd.Timestamp('2019-01-01 16:14:00'),
                    pd.Timestamp('2019-01-01 18:32:00'),
                    pd.Timestamp('2019-01-01 18:44:00')
                ],
                'cash_in': [8, 23, 16, 2]
            }
        )
        result = resample(df)
        expected = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-01-01 16:00:00'),
                    pd.Timestamp('2019-01-01 17:00:00'),
                    pd.Timestamp('2019-01-01 18:00:00'),
                ],
                'cash_in': [31, 0, 18]
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    @patch('foodcast.domain.transform.resample')
    @patch('foodcast.domain.transform.merge')
    @patch('foodcast.domain.transform.clean')
    @patch('foodcast.domain.transform.extract')
    def test_etl(
        self,
        mock_extract: MagicMock,
        mock_clean: MagicMock,
        mock_merge: MagicMock,
        mock_resample: MagicMock
    ) -> None:
        etl('', 4, 6)
        assert mock_extract.call_count == 2
        assert mock_clean.call_count == 2
        mock_merge.assert_called_once()
        mock_resample.assert_called_once()


if __name__ == '__main__':
    unittest.main()
