import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from foodcast.domain.decorators import log_return_shape
logger = logging.getLogger(__name__)


def compute_maes(y_true: pd.Series, y_pred: pd.DataFrame) -> List[float]:
    """
    Return mean absolute errors of multi model given its predictions.

    Parameters
    ----------
    y_pred : pd.DataFrame
        Dataframe containing predicted labels (columns 'y_pred*').

    Returns
    -------
    list
        List of MAEs, one entry per model perturbation.
    """
    columns = [col for col in y_pred.columns if col.startswith('y_pred')]
    return [mean_absolute_error(y_true, y_pred[col]) for col in columns]


def cross_validate(
    model: BaseEstimator,
    x: pd.DataFrame,
    y: pd.DataFrame,
    n_fold: int = 10
) -> Tuple[np.array, pd.DataFrame]:
    """
    Custom cross-validation, compatible with a sklearn TimeSeriesSplit.
    Return MAEs (Mean Absolute Errors) as well as a dataframe of predictions.

    Parameters
    ----------
    model : BaseEstimator
        Model to cross-validate.
    x : pd.DataFrame
        Input features of the training set.
    y : pd.DataFrame
        Input labels of the training set.
    n_fold : int
        Number of temporal cross-validation folds, by default 10.

    Returns
    -------
    Tuple[np.array, pd.DataFrame]
        maes: list of cross-validation MAEs (Mean Absolute Errors).
        preds: pd.DataFrame with one column 'y_true' and one or several columns 'y_pred'.
    """
    maes = []
    preds = pd.DataFrame()
    cv = TimeSeriesSplit(n_fold)
    for fold, (train_index, test_index) in enumerate(cv.split(x, y)):
        x_fold_train, x_fold_test = x.iloc[train_index], x.iloc[test_index]
        y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index]
        model_fold = clone(model)
        model_fold.fit(x_fold_train, y_fold_train)
        try:
            preds_fold_test = model_fold.predict(None, x_fold_test)
        except (TypeError, ValueError):
            preds_fold_test = pd.DataFrame(
                model_fold.predict(x_fold_test),
                index=x_fold_test.index,
                columns=['y_pred_simple']
            )
        mae_fold = compute_maes(y_fold_test, preds_fold_test)
        maes.append(mae_fold)
        preds = pd.concat([preds, preds_fold_test], sort=True)
        logger.info(f'Fold {fold} - train shape: [{x_fold_train.shape} - test shape: {x_fold_test.shape}]')
    maes = np.stack(maes, axis=0)
    return maes, preds


@log_return_shape
def span_future(start: pd.Timestamp, delta: str = '1W', freq: str = '1H') -> pd.DataFrame:
    """
    Generate a dataframe of dates to predict on.
    The future always begins at midnight (after start).
    The future always ends before midnight (after start + delta).

    Parameters
    ----------
    start : pd.Timestamp
        Starting timestamp to predict after.
    delta : str
        Time offset to add from start, by default '1W' (one week).
    freq : str
        New dates frequency sampling, by default '1H4 (one hour).

    Returns
    -------
    pd.DataFrame
        One-column dataframe with date to predict on.
    """
    start = start + pd.Timedelta('1D')
    start = start.normalize()
    end = start + pd.Timedelta(delta) - pd.Timedelta(freq)
    future = pd.date_range(start, end, freq=freq)
    future = future.to_frame(name='order_date')
    future = future.reset_index(drop=True)
    return future


def plotly_predictions(preds: pd.DataFrame, y: Optional[pd.Series] = None) -> go.Figure:
    """
    (Plotly) Plot predictions and true labels if any.

    Parameters
    ----------
    preds : pd.DataFrame
        Predictions.
    y : Optional[pd.Series]
        True labels, by default None.

    Returns
    -------
    go.Figure
        The figure to plot.
    """
    fig = go.Figure()
    columns = [col for col in preds.columns if col.startswith('y_pred')]
    mini = preds[columns].min(axis=1)
    maxi = preds[columns].max(axis=1)
    if 'y_pred_simple' in preds.columns:
        fig.add_trace(
            go.Scatter(
                x=preds.index,
                y=preds['y_pred_simple'],
                line_color='red',
                name='simple predictions'
            )
        )
    if len(columns) > 1:
        fig.add_trace(
            go.Scatter(
                x=mini.index,
                y=mini,
                fill=None,
                line_color='orange',
                line_width=0,
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=maxi.index,
                y=maxi,
                fill='tonexty',
                line_color='orange',
                line_width=0,
                name='multiple predictions'
            )
        )
    if y is not None:
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=y,
                line_color='dodgerblue',
                name='cash-in'
            )
        )
        logger.info(f'plotly_predictions: target shape = {y.shape}')
    logger.info(f'plotly_predictions: predictions shape = {preds.shape}')
    fig.update_layout(
        title='Food forecasting',
        xaxis_title='date',
        yaxis_title='dollars',
        font=dict(
            family="Computer Modern",
            size=18,
            color="#7f7f7f"
        )
    )
    return fig
