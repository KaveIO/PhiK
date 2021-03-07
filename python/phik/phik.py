"""Project: PhiK - correlation analyzer library

Created: 2018/09/05

Description:
    Functions for the Phik correlation calculation

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""
from typing import Tuple, Union, Optional

import numpy as np
import itertools
import pandas as pd
from joblib import Parallel, delayed

from phik import definitions as defs
from .bivariate import phik_from_chi2
from .statistics import get_chi2_using_dependent_frequency_estimates, estimate_simple_ndof
from .binning import create_correlation_overview_table, bin_data
from .data_quality import dq_check_nunique_values, dq_check_hist2d
from .utils import array_like_to_dataframe, guess_interval_cols


def spark_phik_matrix_from_hist2d_dict(spark_context, hist_dict: dict):
    """Correlation matrix of bivariate gaussian using spark parallelization over variable-pair 2d histograms

    See spark notebook phik_tutorial_spark.ipynb as example.

    Each input histogram gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records.
    Correlation coefficient value is between 0 and 1.

    :param spark_context: spark context
    :param hist_dict: dict of 2d numpy grids with value-counts. keys are histogram names.
    :return: phik correlation matrix
    """

    for k, v in hist_dict.items():
        if not isinstance(v, np.ndarray):
            raise TypeError('hist_dict should be a dictionary of 2d numpy arrays.')

    hist_list = list(hist_dict.items())
    hist_rdd = spark_context.parallelize(hist_list)
    phik_rdd = hist_rdd.map(_phik_from_row)
    phik_list = phik_rdd.collect()
    phik_overview = create_correlation_overview_table(phik_list)
    return phik_overview


def _phik_from_row(row: Tuple[str, np.ndarray]) -> Tuple[str, str, float]:
    """Helper function for spark parallel processing

    :param row: rdd row, where row[0] is key and rdd[1]
    :return: union of key, phik-value
    """

    key, grid = row
    c = key.split(':')
    if len(c) == 2 and c[0] == c[1]:
        return c[0], c[1], 1.0
    try:
        phik_value = phik_from_hist2d(grid)
    except TypeError:
        phik_value = np.nan
    return c[0], c[1], phik_value


def phik_from_hist2d(observed:np.ndarray, noise_correction:bool=True) -> float:
    """
    correlation coefficient of bivariate gaussian derived from chi2-value
    
    Chi2-value gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records. 
    Correlation coefficient value is between 0 and 1.

    Bivariate gaussian's range is set to [-5,5] by construction.

    :param observed: 2d-array observed values
    :param bool noise_correction: apply noise correction in phik calculation
    :returns float: correlation coefficient phik
    """

    # chi2 contingency test
    chi2 = get_chi2_using_dependent_frequency_estimates(observed, lambda_='pearson')

    # noise pedestal 
    endof = estimate_simple_ndof(observed) if noise_correction else 0
    pedestal = endof
    if pedestal < 0:
        pedestal = 0

    # phik calculation adds noise pedestal to theoretical chi2
    return phik_from_chi2(chi2, observed.sum(), *observed.shape, pedestal=pedestal)


def phik_from_rebinned_df(data_binned: pd.DataFrame, noise_correction:bool=True, dropna:bool=True,
                          drop_underflow:bool=True, drop_overflow:bool=True) -> pd.DataFrame:
    """
    Correlation matrix of bivariate gaussian derived from chi2-value

    Chi2-value gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records.
    Correlation coefficient value is between 0 and 1.

    Bivariate gaussian's range is set to [-5,5] by construction.

    :param pd.DataFrame data_binned: input data where interval variables have been binned
    :param bool noise_correction: apply noise correction in phik calculation
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :return: phik correlation matrix
    """

    if not dropna:
        # if not dropna replace the NaN values with the string NaN. Otherwise the rows with NaN are dropped
        # by the groupby.
        data_binned.replace(np.nan, defs.NaN, inplace=True)
    if drop_underflow:
        data_binned.replace(defs.UF, np.nan, inplace=True)
    if drop_overflow:
        data_binned.replace(defs.OF, np.nan, inplace=True)

    # cache column order (https://github.com/KaveIO/PhiK/issues/1)
    column_order = data_binned.columns
    
    from phik.config import ncores as NCORES
    if NCORES == 1:
        # Useful when for instance using cProfiler: https://docs.python.org/3/library/profile.html
        phik_list = [
            _calc_phik(co, data_binned[list(co)], noise_correction)
            for co in itertools.combinations_with_replacement(data_binned.columns.values, 2)
        ]
    else:
        phik_list = Parallel(n_jobs=NCORES)(
            delayed(_calc_phik)(co, data_binned[list(co)], noise_correction)
            for co in itertools.combinations_with_replacement(data_binned.columns.values, 2)
        )

    if len(phik_list) == 0:
        return pd.DataFrame(np.nan, index=column_order, columns=column_order)

    phik_overview = create_correlation_overview_table(phik_list)

    # restore column order
    phik_overview = phik_overview.reindex(columns=column_order)
    phik_overview = phik_overview.reindex(index=column_order)

    return phik_overview


def _calc_phik(comb: tuple, data_binned: pd.DataFrame, noise_correction: bool) -> Tuple[str, str, float]:
    """Split off calculation of phik for parallel processing

    :param tuple comb: union of two string columns
    :param pd.DataFrame data_binned: input data where interval variables have been binned
    :param bool noise_correction: apply noise correction in phik calculation
    :return:
    """
    c0, c1 = comb
    if c0 == c1:
        return c0, c1, 1.0

    datahist = data_binned.groupby([c0, c1])[c0].count().to_frame().unstack().fillna(0)

    # If 0 or only 1 values for one of the two variables, it is not possible to calculate phik.
    # This check needs to be done after creation of OF, UF and NaN bins.
    if any([v in datahist.shape for v in [0, 1]]):
        return c0, c1, np.nan

    datahist.columns = datahist.columns.droplevel()
    phikvalue = phik_from_hist2d(datahist.values, noise_correction=noise_correction)
    return c0, c1, phikvalue


def phik_matrix(df:pd.DataFrame, interval_cols:Optional[list]=None, bins:Union[int,list,np.ndarray,dict]=10, quantile:bool=False,
                noise_correction:bool=True, dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True,
                verbose:bool=True) -> pd.DataFrame:
    """
    Correlation matrix of bivariate gaussian derived from chi2-value

    Chi2-value gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records.
    Correlation coefficient value is between 0 and 1.

    Bivariate gaussian's range is set to [-5,5] by construction.

    :param pd.DataFrame data_binned: input data
    :param list interval_cols: column names of columns with interval variables.
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param quantile: when bins is an integer, uniform bins (False) or bins based on quantiles (True)
    :param bool noise_correction: apply noise correction in phik calculation
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :param bool verbose: if False, do not print all interval columns that are guessed
    :return: phik correlation matrix
    """

    if interval_cols is None:
        interval_cols = guess_interval_cols(df, verbose)

    df_clean, interval_cols_clean = dq_check_nunique_values(df, interval_cols, dropna=dropna)

    data_binned, binning_dict = bin_data(df_clean, cols=interval_cols_clean, bins=bins, quantile=quantile, retbins=True)

    return phik_from_rebinned_df(
        data_binned, noise_correction, dropna=dropna, drop_underflow=drop_underflow, drop_overflow=drop_overflow
    )


def global_phik_from_rebinned_df(data_binned:pd.DataFrame, noise_correction:bool=True, dropna:bool=True,
                                 drop_underflow:bool=True, drop_overflow:bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Global correlation values of bivariate gaussian derived from chi2-value from rebinned df

    Chi2-value gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records.
    Correlation coefficient value is between 0 and 1.

    Bivariate gaussian's range is set to [-5,5] by construction.

    :param pd.DataFrame data_binned: rebinned input data
    :param bool noise_correction: apply noise correction in phik calculation
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :return: global correlations array
    """

    phik_overview = phik_from_rebinned_df(
        data_binned, noise_correction, dropna=dropna, drop_underflow=drop_underflow, drop_overflow=drop_overflow
    )
    from numpy.linalg import inv
    V = phik_overview.values
    Vinv = inv(V)
    global_correlations = np.array([[np.sqrt(1 - 1/(V[i][i] * Vinv[i][i]))] for i in range(V.shape[0])])
    return global_correlations, phik_overview.index.values


def global_phik_array(df:pd.DataFrame, interval_cols:list=None, bins:Union[int,list,np.ndarray,dict]=10, quantile:bool=False,
                      noise_correction:bool=True, dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True,
                      verbose:bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Global correlation values of bivariate gaussian derived from chi2-value

    Chi2-value gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records.
    Correlation coefficient value is between 0 and 1.

    Bivariate gaussian's range is set to [-5,5] by construction.

    :param pd.DataFrame data_binned: input data
    :param list interval_cols: column names of columns with interval variables.
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param quantile: when bins is an integer, uniform bins (False) or bins based on quantiles (True)
    :param bool noise_correction: apply noise correction in phik calculation
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :param bool verbose: if False, do not print all interval columns that are guessed
    :return: global correlations array
    """

    if interval_cols is None:
        interval_cols = guess_interval_cols(df, verbose)

    df_clean, interval_cols_clean = dq_check_nunique_values(df, interval_cols, dropna=dropna)

    data_binned, binning_dict = bin_data(df_clean, cols=interval_cols_clean, bins=bins, quantile=quantile, retbins=True)
    return global_phik_from_rebinned_df(
        data_binned, noise_correction=noise_correction, dropna=dropna, drop_underflow=drop_underflow,
        drop_overflow=drop_overflow
    )


def phik_from_array(x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series], num_vars: Union[str, list]=None,
                    bins:Union[int, dict, list, np.ndarray]=10, quantile:bool=False, noise_correction:bool=True, dropna:bool=True,
                    drop_underflow:bool=True, drop_overflow:bool=True) -> float:
    """
    Correlation matrix of bivariate gaussian derived from chi2-value

    Chi2-value gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records.
    Correlation coefficient value is between 0 and 1.

    Bivariate gaussian's range is set to [-5,5] by construction.

    :param x: array-like input
    :param y: array-like input
    :param num_vars: list of numeric variables which need to be binned, e.g. ['x'] or ['x','y']
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param quantile: when bins is an integer, uniform bins (False) or bins based on quantiles (True)
    :param bool noise_correction: apply noise correction in phik calculation
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :return: phik correlation coefficient
    """
    if num_vars is None:
        num_vars = []
    elif isinstance(num_vars, str):
        num_vars = [num_vars]

    if len(num_vars) > 0:
        df = array_like_to_dataframe(x, y)
        x, y = bin_data(df, num_vars, bins=bins, quantile=quantile).T.values

    return phik_from_binned_array(
        x, y, noise_correction=noise_correction, dropna=dropna, drop_underflow=drop_underflow, drop_overflow=drop_overflow
    )


def phik_from_binned_array(x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series], noise_correction:bool=True, dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True) -> float:
    """
    Correlation matrix of bivariate gaussian derived from chi2-value

    Chi2-value gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records.
    Correlation coefficient value is between 0 and 1.

    Bivariate gaussian's range is set to [-5,5] by construction.

    :param x: array-like input. Interval variables need to be binned beforehand.
    :param y: array-like input. Interval variables need to be binned beforehand.
    :param bool noise_correction: apply noise correction in phik calculation
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :return: phik correlation coefficient
    """

    if not dropna:
        x = pd.Series(x).fillna(defs.NaN).astype(str).values
        y = pd.Series(y).fillna(defs.NaN).astype(str).values

    if drop_underflow or drop_overflow:
        x = pd.Series(x).astype(str).values
        y = pd.Series(y).astype(str).values
        if drop_underflow:
            x[np.where(x == defs.UF)] = np.nan
            y[np.where(y == defs.UF)] = np.nan
        if drop_overflow:
            y[np.where(y == defs.OF)] = np.nan
            x[np.where(x == defs.OF)] = np.nan

    hist2d = pd.crosstab(x, y).values

    if not dq_check_hist2d(hist2d):
        return np.nan

    return phik_from_hist2d(hist2d, noise_correction=noise_correction)
