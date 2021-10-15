from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from typing import Optional, Any
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from foodcast.domain.forecast import compute_maes, cross_validate, span_future, plotly_predictions


class DummyModel(BaseEstimator, RegressorMixin):  # type: ignore

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> DummyModel:
        self.estimator.fit(X, np.ravel(y))
        return self

    def predict(self, context: Any, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.estimator.predict(X),
            index=X.index,
            columns=['y_pred']
        )


class TestForecast(unittest.TestCase):

    def test_compute_maes(self) -> None:
        y_pred = pd.DataFrame(
            {
                'y_pred_0': [1.0, 2.0, 3.0],
                'y_pred_1': [0.0, 2.0, 4.0]
            }
        )
        y_true = pd.Series([1.0, 2.0, 3.5])
        maes_result = compute_maes(y_true, y_pred)
        maes_expected = [0.1666666, 0.5]
        np.testing.assert_almost_equal(maes_result, maes_expected)

    def test_cross_validate_1(self) -> None:
        X = pd.DataFrame(
            {
                'X1': [1, 2, 3, 4],
                'X2': [1, 2, 5, 6]
            },
            index=[
                pd.Timestamp('2019-10-08 14:00:00'),
                pd.Timestamp('2019-10-08 15:00:00'),
                pd.Timestamp('2019-10-08 16:00:00'),
                pd.Timestamp('2019-10-08 17:00:00')
            ]
        )
        y = pd.DataFrame({'y': [3.0, 6.0, -11.0, 2.0]}, index=X.index)
        model = DummyModel(DecisionTreeRegressor(random_state=1))
        maes_result, preds_result = cross_validate(model, X, y, n_fold=2)
        maes_expected = np.array([[17], [13]])
        preds_expected = pd.DataFrame(
            {
                'y_pred': [6.0, -11.0],
            },
            index=[
                pd.Timestamp('2019-10-08 16:00:00'),
                pd.Timestamp('2019-10-08 17:00:00')
            ]
        )
        np.testing.assert_almost_equal(maes_result, maes_expected)
        pd.testing.assert_frame_equal(preds_result, preds_expected)

    def test_cross_validate_2(self) -> None:
        X = pd.DataFrame(
            {
                'X1': [1, 2, 3, 4],
                'X2': [1, 2, 5, 6]
            },
            index=[
                pd.Timestamp('2019-10-08 14:00:00'),
                pd.Timestamp('2019-10-08 15:00:00'),
                pd.Timestamp('2019-10-08 16:00:00'),
                pd.Timestamp('2019-10-08 17:00:00')
            ]
        )
        y = pd.DataFrame({'y': [3.0, 6.0, -11.0, 2.0]}, index=X.index)
        model = DecisionTreeRegressor(random_state=1)
        maes_result, preds_result = cross_validate(model, X, y, n_fold=2)
        maes_expected = np.array([[17], [13]])
        preds_expected = pd.DataFrame(
            {
                'y_pred_simple': [6.0, -11.0],
            },
            index=[
                pd.Timestamp('2019-10-08 16:00:00'),
                pd.Timestamp('2019-10-08 17:00:00')
            ]
        )
        np.testing.assert_almost_equal(maes_result, maes_expected)
        pd.testing.assert_frame_equal(preds_result, preds_expected)

    def test_span_future(self) -> None:
        start = pd.Timestamp('2019-10-08 21:00:00')
        result = span_future(start, delta='1D', freq='1H')
        expected = pd.DataFrame(
            {
                'order_date': pd.date_range('2019-10-09 00:00:00', '2019-10-09 23:00:00', freq='1H')
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_plotly_predictions_1(self) -> None:
        preds = pd.DataFrame({'y_pred_simple': [1, 2, 3, 4]}, index=[55, 56, 57, 58])
        try:
            plotly_predictions(preds)
        except Exception:
            self.fail()

    def test_plotly_predictions_2(self) -> None:
        y = pd.Series([1, 2, 3, 4], index=[55, 56, 57, 58])
        preds = pd.DataFrame(
            {
                'y_pred_0': [1, 2, 3, 4],
                'y_pred_1': [-1, 3, 2, 6]
            },
            index=[55, 56, 57, 58]
        )
        try:
            plotly_predictions(preds, y)
        except Exception:
            self.fail()
