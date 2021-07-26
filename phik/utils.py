from typing import Union

import pandas as pd
import numpy as np


def array_like_to_dataframe(
    x: Union[pd.Series, list, np.ndarray], y: Union[pd.Series, list, np.ndarray]
) -> pd.DataFrame:
    """Concat two array-like data structures into a DataFrame

    :param x: pd.Series, list or np.ndarray
    :param y: pd.Series, list or np.ndarray
    :return: pd.DataFrame
    """
    x_name = x.name if isinstance(x, pd.Series) else "x"
    y_name = y.name if isinstance(y, pd.Series) else "y"

    return pd.DataFrame(np.array([x, y]).T, columns=[x_name, y_name])


def guess_interval_cols(df: pd.DataFrame, verbose: bool = False) -> list:
    """Select columns that have a dtype part of np.number

    :param df: DataFrame
    :param bool verbose: print all interval columns that are guessed
    :return: list of interval columns
    """
    interval_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if interval_cols and verbose:
        print("interval columns not set, guessing: {}".format(str(interval_cols)))

    if not isinstance(interval_cols, list):
        raise ValueError("Could not guess interval columns")
    return interval_cols


def make_shapes_equal(observed: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Make observed and expected (pandas) histograms equal in shape

    Sometimes expected histogram shape need filling / pruning to make its shape equal to observed. Give expected the
    same number of columns and rows. Needed for proper histogram comparison.

    :param pd.DataFrame observed: The observed contingency table. The table contains the observed number of occurrences in each cell.
    :param pd.DataFrame expected: The expected contingency table. The table contains the expected number of occurrences in each cell.
    :return: expected frequencies, having the same shape as observed
    """
    # columns
    o_cols = observed.columns.tolist()
    e_cols = expected.columns.tolist()
    o_cols_missing = list(set(e_cols) - set(o_cols))
    e_cols_missing = list(set(o_cols) - set(e_cols))
    # index
    o_idx = observed.index.tolist()
    e_idx = expected.index.tolist()
    o_idx_missing = list(set(e_idx) - set(o_idx))
    e_idx_missing = list(set(o_idx) - set(e_idx))

    # make expected columns equal to observed
    for c in o_cols_missing:
        observed[c] = 0.0
    for c in e_cols_missing:
        expected[c] = 0.0
    observed.columns = sorted(observed.columns)
    expected.columns = sorted(expected.columns)
    # this should always be a match now
    assert len(observed.columns) == len(expected.columns)

    # make expected index equal to observed
    for i in o_idx_missing:
        observed.loc[i] = np.zeros(len(observed.columns))
    for i in e_idx_missing:
        expected.loc[i] = np.zeros(len(expected.columns))
    # this should always be a match now
    assert len(observed.index) == len(expected.index)

    return expected
