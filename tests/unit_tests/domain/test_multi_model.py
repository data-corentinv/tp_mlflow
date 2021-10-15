import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from foodcast.domain.multi_model import MultiModel


class TestMultiModel(unittest.TestCase):

    def test_init(self) -> None:
        model = MultiModel(estimator='estimator', n_models=10)
        assert hasattr(model, 'estimator')
        assert hasattr(model, 'n_models')
        assert model.estimator == 'estimator'
        assert model.n_models == 10

    def test_fit_1(self) -> None:
        X = pd.DataFrame(
            {
                'X1': [1, 2, 3, 4],
                'X2': [1, 2, 5, 6],
            }
        )
        y = pd.DataFrame({'y': [3, 6, 13, 16]})
        model = MultiModel(RandomForestRegressor(n_estimators=10), n_models=2)
        model.fit(X, y)
        check_is_fitted(model, ['single_estimator', 'estimators'])
        assert len(model.estimators) == 2
        assert model.estimators[0].random_state == 0
        assert model.estimators[1].random_state == 1
        attributes = ['estimators_', 'feature_importances_', 'n_features_', 'n_outputs_']
        check_is_fitted(model.estimators[0], attributes)
        check_is_fitted(model.estimators[1], attributes)
        assert model.estimators[0].n_features_ == 2
        assert model.estimators[1].n_features_ == 2
        assert model.estimators[0].n_outputs_ == 1
        assert model.estimators[1].n_outputs_ == 1
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            model.estimators[0].feature_importances_,
            model.estimators[1].feature_importances_
        )

    def test_fit_2(self) -> None:
        X = pd.DataFrame(
            {
                'X1': [1, 2, 3, 4],
                'X2': [1, 2, 5, 6],
            }
        )
        y = pd.DataFrame({'y': [3, 6, 13, 16]})
        model = MultiModel(LinearRegression(), n_models=2)
        model.fit(X, y)
        check_is_fitted(model, ['single_estimator', 'estimators'])
        assert len(model.estimators) == 2
        attributes = ['coef_', 'intercept_']
        check_is_fitted(model.estimators[0], attributes)
        check_is_fitted(model.estimators[1], attributes)
        assert len(model.estimators[0].coef_) == 2
        assert len(model.estimators[1].coef_) == 2

    def test_predict(self) -> None:
        X_train = pd.DataFrame(
            {
                'X1': [1, 2, 3, 4],
                'X2': [1, 2, 5, 6],
            }
        )
        y_train = pd.DataFrame({'y': [3, 8, -13, -16]})
        model = MultiModel(RandomForestRegressor(n_estimators=10, random_state=42), n_models=2)
        model.fit(X_train, y_train)
        X_pred = pd.DataFrame(
            {
                'X1': [-1, 1, 3, 4],
                'X2': [-1, -1, 1, 7],
            }
        )
        y_result = model.predict(None, X_pred)
        y_expected = pd.DataFrame(
            {
                'y_pred_0': [3.5, 3.5, 1.2, 0.0],
                'y_pred_1': [3.5, 3.5, 2.6, 0.0],
                'y_pred_simple': [1.3, 1.3, -1.4, -12.7]
            },
        )
        pd.testing.assert_frame_equal(y_result, y_expected)
