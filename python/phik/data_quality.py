"""Project: PhiK - correlation analyzer library

Created: 2018/12/28

Description:
    A set of functions to check for data quality issues in input data.

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import warnings
import copy
import pandas as pd


def dq_check_nunique_values(df, interval_cols, dropna=True):

    """
    Basic data quality checks per column in a dataframe.

    The following checks are done:

    1. For all non-interval variables, if the number of unique values per variable is larger than 100 a warning is printed.
    When the number of unique values is large, the variable is likely to be an interval variable. Calculation of phik
    will be slow(ish) for pairs of variables where one (or two) have many different values (i.e. many bins).

    2. For all interval variables, the number of unique values must be at least two. If the number of unique values is
    zero (i.e. all NaN) the column is removed. If the number of unique values is one, it is not possible to
    automatically create a binning for this variable (as min and max are the same). The variable is therefore dropped,
    irrespective of whether dropna is True or False.

    3. For all non-interval variables, the number of unique values must be at least either
    a) 1 if dropna=False (NaN is now also considered a valid category), or
    b) 2 if dropna=True

    The function returns a dataframe where all columns with invalid data are removed. Also the list of interval_cols
    is updated and returned.

    :param pd.DataFrame df: input data
    :param list interval_cols: column names of columns with interval variables.
    :param bool dropna: remove NaN values when True
    :returns: cleaned data, updated list of interval columns
    """

    # check non-interval variable for number of unique values
    for col in sorted(list(set(df.columns)-set(interval_cols))):
        if df[col].nunique() > 100:
            warnings.warn('The number of unique values of variable {0:s} is very large: {1:d}. Are you sure this is '
                          'not an interval variable? Analysis for pairs of variables including {0:s} might be slow.'
                          .format(col, df[col].nunique()))

    drop_cols = []

    # check for interval values whether there are at least two unique values (otherwise I cannot bin automatically)
    for col in interval_cols:
        if df[col].nunique() < 2:
            drop_cols.append(col)
            warnings.warn('Not enough unique value for variable {0:s} for analysis {1:d}. Dropping this column'
                          .format(col, df[col].nunique()))

    # check non-interval values whether there are at least two different values OR 1 value and NaN if dropna==False
    for col in sorted(list(set(df.columns) - set(interval_cols))):
        if df[col].nunique() == 0 or (df[col].nunique() == 1 and dropna):
            drop_cols.append(col)
            warnings.warn('Not enough unique value for variable {0:s} for analysis {1:d}. Dropping this column'
                          .format(col, df[col].nunique()))

    df_clean = df.copy()
    interval_cols_clean = copy.copy(interval_cols)
    if len(drop_cols) > 0:
        cols = sorted(list(set(df.columns) - set(drop_cols)))
        interval_cols_clean = sorted(list(set(interval_cols) - set(drop_cols)))

        df_clean = df_clean[cols]

    return df_clean, interval_cols_clean


def dq_check_hist2d(hist2d):

    """Basic data quality checks for a contingency table

    The Following checks are done:

    1. There must be at least two bins in both the x and y direction.

    2. If the number of bins in the x and/or y direction is larger than 100 a warning is printed.

    :param hist2d: contigency table
    :return: bool passed_check
    """

    if 0 in hist2d.shape or 1 in hist2d.shape:
        warnings.warn('Too few unique values for variable x ({0:d}) or y ({1:d})'.format(
            hist2d.shape[0], hist2d.shape[1]))
        return False
    if hist2d.shape[0] > 100:
        warnings.warn('The number of unique values of variable x is large: {0:d}. Are you sure this is '
                      'not an interval variable? Analysis might be slow.'
                      .format(hist2d.shape[0]))
    if hist2d.shape[1] > 100:
        warnings.warn('The number of unique values of variable y is large: {0:d}. Are you sure this is '
                      'not an interval variable? Analysis might be slow.'
                      .format(hist2d.shape[0]))

    return True
