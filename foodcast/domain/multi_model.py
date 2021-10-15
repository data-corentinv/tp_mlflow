from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Optional, Any
from mlflow.pyfunc import PythonModel
from sklearn.utils import check_X_y, check_array, resample
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
logger = logging.getLogger(__name__)


class MultiModel(PythonModel, BaseEstimator, RegressorMixin):  # type: ignore
    """
    Wrapper of multiple clones of a given estimator. Each clone differs only by:
        - the boostrap sample it is trained on
        - its random state (if any).
    Inherits from PythonModel so as to be saved with MLflow.
    Inherits BaseEstimator and RegressorMixin so as to fit into
    sklearn pipelines and sklearn clone method.

    Attributes
    ----------
    estimator : sklearn.BaseEstimator
        Any scikit-learn estimator.
    n : int
        Number of perturbed estimators.
    estimators : list
        List of fitted estimators.
    """

    def __init__(self, estimator: Optional[BaseEstimator] = None, n_models: int = 10) -> None:
        """
        Initialize the wrapper model.

        Parameters
        ----------
        estimator : BaseEstimator, optional
            Any sklearn model having a random_state attribute, by default None.
        n : int, optional
            Number of clones to maintain, by default 10.
        """
        self.n_models = n_models
        self.estimator = estimator
        logger.info(f'Instantiate {n_models} models of type:\n{estimator}')

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> MultiModel:
        """
        Fit all clones and rearrange them into a list.
        The initial estimator is fit apart.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.
        y : Optional[pd.Series] of shape (n_samples,)
            Training labels, by default None.

        Returns
        -------
        MultiModel
            The model itself.
        """
        X, y = check_X_y(X, np.ravel(y))
        self.single_estimator = clone(self.estimator)
        self.single_estimator.fit(X, y)
        self.estimators = []
        for random_state in range(self.n_models):
            e = clone(self.estimator)
            if hasattr(e, 'random_state'):
                e.set_params(random_state=random_state)
            X_bootstrap, y_bootstrap = resample(X, y, replace=True, random_state=random_state)
            e.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(e)
            logger.info(f'fit: X of shape {X.shape} on y - seed: {random_state}')
        return self

    def predict(self, context: Any, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for each clone and concatenate the results into a pandas dataframe.
        Predictions are bounded above zero.

        Parameters
        ----------
        context : Any
            Used by MLflow in some cases.
        X : pd.DataFrame of shape (n_samples, n_features)
            Prediction data.

        Returns
        -------
        pd.DataFrame
            Concatenation of each clone predictions.
        """
        check_is_fitted(self, ["single_estimator", "estimators"])
        X_index = X.index
        X = check_array(X)
        preds = np.stack([e.predict(X) for e in self.estimators], axis=1)
        preds = np.maximum(0, preds)
        preds = pd.DataFrame(
            preds,
            index=X_index,
            columns=['y_pred_{}'.format(i) for i in range(self.n_models)]
        )
        preds['y_pred_simple'] = self.single_estimator.predict(X)
        logger.info(f'predict: X of shape {X.shape}')
        return preds
