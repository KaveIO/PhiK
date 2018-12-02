"""Project: PhiK - correlation analyzer library

Created: 2018/09/05

Description:
    TODO

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import numpy as np
import itertools
import pandas as pd

from phik import definitions as defs
from .bivariate import phik_from_chi2
from .statistics import get_chi2_using_dependent_frequency_estimates, estimate_simple_ndof
from .binning import create_correlation_overview_table, bin_data


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
    if not isinstance(observed, np.ndarray):
        raise TypeError('observed is not a numpy array.')

    # chi2 contingency test
    chi2 = get_chi2_using_dependent_frequency_estimates(observed, lambda_='pearson')

    # noise pedestal 
    endof = estimate_simple_ndof(observed) if noise_correction else 0
    pedestal = endof
    if pedestal < 0:
        pedestal = 0

    # phik calculation adds noise pedestal to theoretical chi2
    phik = phik_from_chi2(chi2, observed.sum(), *observed.shape, None, None, pedestal)    

    return phik


def phik_from_rebinned_df(data_binned:pd.DataFrame, noise_correction:bool=True, dropna:bool=True,
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
    if not isinstance(data_binned, pd.DataFrame):
        raise TypeError('data_binned is not a pandas DataFrame.')

    if not dropna:
        # if not dropna replace the NaN values with the string NaN. Otherwise the rows with NaN are dropped
        # by the groupby.
        data_binned = data_binned.replace(np.nan, defs.NaN).copy()
    if drop_underflow:
        data_binned = data_binned.replace(defs.UF, np.nan).copy()
    if drop_overflow:
        data_binned = data_binned.replace(defs.OF, np.nan).copy()

    phiks = {}
    for i, comb in enumerate(itertools.combinations_with_replacement(data_binned.columns.values, 2)):
        c0, c1 = comb
        if c0 == c1:
            phiks[':'.join(comb)] = 1
            continue
        datahist = data_binned.groupby([c0, c1])[c0].count().to_frame().unstack().fillna(0)
        datahist.columns = datahist.columns.droplevel()
        phikvalue = phik_from_hist2d(datahist.values, noise_correction=noise_correction)
        phiks[':'.join(comb)] = phikvalue

    phik_overview = create_correlation_overview_table(phiks)
    return phik_overview


def phik_matrix(df:pd.DataFrame, interval_cols:list=None, bins=10, quantile:bool=False, noise_correction:bool=True,
                dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True) -> pd.DataFrame:
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
    :return: phik correlation matrix
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df is not a pandas DataFrame.')
    if not isinstance(bins, (int,list,np.ndarray,dict)):
        raise TypeError('bins is of incorrect type.')    

    if isinstance( interval_cols, type(None) ):
        interval_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if interval_cols:
            print('interval_cols not set, guessing: {0:s}'.format(str(interval_cols)))
    assert isinstance( interval_cols, list ), 'interval_cols is not a list.'

    data_binned, binning_dict = bin_data(df, cols=interval_cols, bins=bins, quantile=quantile, retbins=True)
    return phik_from_rebinned_df(data_binned, noise_correction, dropna=dropna, drop_underflow=drop_underflow,
                                 drop_overflow=drop_overflow)


def global_phik_from_rebinned_df(data_binned:pd.DataFrame, noise_correction:bool=True,
                                 dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True) -> pd.DataFrame:
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
    if not isinstance(data_binned, pd.DataFrame):
        raise TypeError('data_binned is not a pandas DataFrame.')

    phik_overview = phik_from_rebinned_df(data_binned, noise_correction, dropna=dropna, drop_underflow=drop_underflow, \
                                          drop_overflow=drop_overflow)
    from numpy.linalg import inv
    V = phik_overview.values
    Vinv = inv(V)
    global_correlations = np.array([[np.sqrt(1 - 1/(V[i][i] * Vinv[i][i]))] for i in range(V.shape[0]) ])
    return global_correlations, phik_overview.index.values


def global_phik_array(df:pd.DataFrame, interval_cols:list=None, bins=10, quantile:bool=False, noise_correction:bool=True,
                      dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True) -> pd.DataFrame:
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
    :return: global correlations array
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df is not a pandas DataFrame.')
    if not isinstance(bins, (int,list,np.ndarray,dict)):
        raise TypeError('bins is of incorrect type.')    

    if isinstance( interval_cols, type(None) ):
        interval_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if interval_cols:
            print('interval_cols not set, guessing: {0:s}'.format(str(interval_cols)))
    assert isinstance( interval_cols, list ), 'interval_cols is not a list.'

    data_binned, binning_dict = bin_data(df, cols=interval_cols, bins=bins, quantile=quantile, retbins=True)
    return global_phik_from_rebinned_df(data_binned, noise_correction=noise_correction, dropna=dropna, \
                                        drop_underflow=drop_underflow, drop_overflow=drop_overflow)


def phik_from_array(x, y, num_vars:list=[], bins=10, quantile:bool=False, noise_correction:bool=True, dropna:bool=True,
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
    if not isinstance(x, (np.ndarray, pd.Series)):
        raise TypeError('x is not array like.')
    if not isinstance(y, (np.ndarray, pd.Series)):
        raise TypeError('y is not array like.')
    if not isinstance(bins, (int,list,np.ndarray,dict)):
        raise TypeError('bins is of incorrect type.')    

    if isinstance(num_vars, str):
        num_vars = [num_vars]

    if len(num_vars) > 0:
        df = pd.DataFrame(np.array([x, y]).T, columns=['x', 'y'])
        x, y = bin_data(df, num_vars, bins=bins, quantile=quantile).T.values

    return phik_from_binned_array(x, y, noise_correction=noise_correction, dropna=dropna,
                                  drop_underflow=drop_underflow, drop_overflow=drop_overflow)


def phik_from_binned_array(x, y, noise_correction:bool=True, dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True) -> float:
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
    if not isinstance(x, (np.ndarray, pd.Series)):
        raise TypeError('x is not array like.')
    if not isinstance(y, (np.ndarray, pd.Series)):
        raise TypeError('y is not array like.')

    if not dropna:
        x = pd.Series(x).fillna(defs.NaN).astype(str).values
        y = pd.Series(y).fillna(defs.NaN).astype(str).values

    if drop_underflow or drop_overflow:
        x = x.copy()
        y = y.copy()
    if drop_underflow:
        x[np.where(x == defs.UF)] = np.nan
        x[np.where(x == defs.OF)] = np.nan
    if drop_overflow:
        y[np.where(y == defs.UF)] = np.nan
        y[np.where(y == defs.OF)] = np.nan

    hist2d = pd.crosstab(x, y).values

    return phik_from_hist2d(hist2d, noise_correction=noise_correction)
