import logging
import pandas as pd
from functools import wraps
from typing import TypeVar, Callable, Any
F = TypeVar('F', bound=Callable[..., Any])


def log_return_shape(func: F) -> F:
    """
    Decorator for printing return shape in case a function return a pandas.DataFrame

    Parameters
    ----------
    func : Callable
        Function to decorate

    Returns
    -------
    Callable
        Decorated function
    """
    @wraps(func)
    def with_shape(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__code__.co_filename)
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            logger.info(f'{func.__name__}: shape = {result.shape}')
        return result
    return with_shape  # type: ignore
