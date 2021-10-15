import unittest
import numpy as np
import pandas as pd
from foodcast.domain.feature_engineering import dummy_day, hour_cos_sin
from foodcast.domain.feature_engineering import lag_offline, lag_online, features_offline, features_online


class TestFeatureEngineering(unittest.TestCase):

    def test_dummy_day(self) -> None:
        df = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-08 17:32:00'),
                    pd.Timestamp('2019-10-09 17:32:00')
                ]
            }
        )
        result = dummy_day(df)
        expected = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-08 17:32:00'),
                    pd.Timestamp('2019-10-09 17:32:00')
                ],
                'day_2': [0, 1],
            }
        )
        expected['day_2'] = expected['day_2'].astype(np.uint8)
        pd.testing.assert_frame_equal(result, expected)

    def test_hour_cos_sin(self) -> None:
        df = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-08 17:32:00'),
                    pd.Timestamp('2019-10-09 08:14:00')
                ]
            }
        )
        result = hour_cos_sin(df)
        expected = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-08 17:32:00'),
                    pd.Timestamp('2019-10-09 08:14:00')
                ],
                'hour_cos_1': [-0.25881904510252063, -0.5000000000000004],
                'hour_sin_1': [-0.9659258262890683, 0.8660254037844384]
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_lag_offline(self) -> None:
        df = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-08 17:00:00'),
                    pd.Timestamp('2019-10-08 18:00:00'),
                    pd.Timestamp('2019-10-15 17:00:00'),
                    pd.Timestamp('2019-10-15 18:00:00')
                ],
                'cash_in': [50.0, 75.0, 60.0, 85.0]
            }
        )
        result = lag_offline(df)
        expected = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-15 17:00:00'),
                    pd.Timestamp('2019-10-15 18:00:00')
                ],
                'cash_in': [60.0, 85.0],
                'lag_1W': [50.0, 75.0]
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_lag_online(self) -> None:
        df = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-14 09:00:00'),
                    pd.Timestamp('2019-10-15 17:00:00'),
                    pd.Timestamp('2019-10-15 18:00:00')
                ]
            }
        )
        past = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-08 17:00:00'),
                    pd.Timestamp('2019-10-08 18:00:00')
                ],
                'cash_in': [50.0, 75.0]
            }
        )
        result = lag_online(df, past)
        expected = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-14 09:00:00'),
                    pd.Timestamp('2019-10-15 17:00:00'),
                    pd.Timestamp('2019-10-15 18:00:00')
                ],
                'lag_1W': [0.0, 50.0, 75.0]
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_features_offline(self) -> None:
        df = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-07 17:00:00'),
                    pd.Timestamp('2019-10-08 17:00:00'),
                    pd.Timestamp('2019-10-08 18:00:00'),
                    pd.Timestamp('2019-10-15 17:00:00'),
                    pd.Timestamp('2019-10-15 18:00:00')
                ],
                'cash_in': [10.0, 25.0, 50.0, 60.0, 85.0]
            }
        )
        result = features_offline(df)
        expected = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-15 17:00:00'),
                    pd.Timestamp('2019-10-15 18:00:00')
                ],
                'cash_in': [60.0, 85.0],
                'day_1': [1, 1],
                'hour_cos_1': [-0.25881904510252063, 0],
                'hour_sin_1': [-0.9659258262890683, -1],
                'lag_1W': [25.0, 50.0]
            }
        )
        expected['day_1'] = expected['day_1'].astype(np.uint8)
        pd.testing.assert_frame_equal(result, expected)

    def test_online(self) -> None:
        df = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-15 17:00:00'),
                    pd.Timestamp('2019-10-15 18:00:00'),
                    pd.Timestamp('2019-10-16 19:00:00')
                ],
            }
        )
        past = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-07 17:00:00'),
                    pd.Timestamp('2019-10-08 17:00:00'),
                    pd.Timestamp('2019-10-08 18:00:00')
                ],
                'cash_in': [10.0, 25.0, 50.0]
            }
        )
        result = features_online(df, past)
        expected = pd.DataFrame(
            {
                'order_date': [
                    pd.Timestamp('2019-10-15 17:00:00'),
                    pd.Timestamp('2019-10-15 18:00:00'),
                    pd.Timestamp('2019-10-16 19:00:00')
                ],
                'day_2': [0, 0, 1],
                'hour_cos_1': [-0.25881904510252063, 0, 0.2588190451025203],
                'hour_sin_1': [-0.9659258262890683, -1, -0.9659258262890684],
                'lag_1W': [25.0, 50.0, 0.0]
            }
        )
        expected['day_2'] = expected['day_2'].astype(np.uint8)
        pd.testing.assert_frame_equal(result, expected)
