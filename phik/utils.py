from typing import Union

import pandas as pd
import numpy as np


def array_like_to_dataframe(x: Union[pd.Series, list, np.ndarray], y: Union[pd.Series, list, np.ndarray]):
    """Concat two array-like data structures into a DataFrame

    :param x: pd.Series, list or np.ndarray
    :param y: pd.Series, list or np.ndarray
    :return: pd.DataFrame
    """
    x_name = x.name if isinstance(x, pd.Series) else 'x'
    y_name = y.name if isinstance(y, pd.Series) else 'y'

    return pd.DataFrame(np.array([x, y]).T, columns=[x_name, y_name])


def guess_interval_cols(df: pd.DataFrame, verbose:bool=False) -> list:
    """Select columns that have a dtype part of np.number

    :param df: DataFrame
    :param bool verbose: print all interval columns that are guessed
    :return: list of interval columns
    """
    interval_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if interval_cols and verbose:
        print('interval columns not set, guessing: {}'.format(str(interval_cols)))

    if not isinstance(interval_cols, list):
        raise ValueError('Could not guess interval columns')
    return interval_cols
