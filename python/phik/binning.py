"""Project: PhiK - correlation analyzer library

Created: 2018/09/06

Description:
    A set of rebinning functions, to help rebin two lists into a 2d histogram.

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import numpy as np
import pandas as pd
import warnings

from phik import definitions as defs


def bin_edges(arr, nbins:int, quantile:bool = False) -> np.ndarray:
    """
    Create uniform or quantile bin-edges for the input array.

    :param arr: array like object with input data
    :param int nbins: the number of bin
    :param bool quantile: uniform bins (False) or bins based on quantiles (True)
    :returns: array with bin edges
    """
    if not isinstance(arr, (np.ndarray, list, pd.Series)):
        raise TypeError('arr is not array like.')

    if quantile:
        quantiles = np.linspace(0, 1, nbins + 1)
        xbins = np.quantile(arr[~np.isnan(arr)], quantiles)
        xbins[0] = xbins[0] - 1E-14
    else:
        xbins = np.linspace(min(arr) - 1E-14, max(arr), nbins + 1)

    return xbins


def bin_array(arr, bin_edges):
    """
    Index the data given the bin_edges. 

    Underflow and overflow values are indicated.

    :param arr: array like object with input data
    :param bin_edges: list with bin edges.
    :returns: indexed data
    """
    if not isinstance(arr, (np.ndarray, list)):
        raise TypeError('arr is not a list or numpy array.')
    if not isinstance(bin_edges, (np.ndarray, list)):
        raise TypeError('bin_edges is not a list or numpy array.')

    # Bin data
    binned_arr = np.searchsorted(bin_edges, arr).astype(object)

    # Check if all bins are filled and store bin-labels
    bin_labels = []
    bin_indices = pd.Series(binned_arr).value_counts().index
    for i in range(1, len(bin_edges)):
        if i not in bin_indices:
            warnings.warn('Empty bin with bin-edges {0:s} - {1:s}'.format(str(bin_edges[i-1]), str(bin_edges[i])))
        else:
            bin_labels.append((bin_edges[i-1], bin_edges[i]))

    # NaN values are added to the overflow bin. Restore NaN values:
    binned_arr[np.argwhere(np.isnan(arr))] = np.nan

    # Set underflow values to UF
    binned_arr[np.argwhere(binned_arr == 0)] = defs.UF

    # Set overflow values to OF
    binned_arr[np.argwhere(binned_arr == len(bin_edges))] = defs.OF

    return binned_arr, bin_labels


def bin_data(data, cols:list=[], bins=10, quantile: bool=False, retbins: bool=False):
    """
    Index the input dataframe given the bin_edges for the columns specified in cols.

    :param DataFrame data: input data
    :param list cols: list of columns with numeric data which needs to be indexed
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param quantile: when bins is an integer, uniform bins (False) or bins based on quantiles (True)
    :returns: rebinned dataframe
    :rtype: pandas.DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data is not a pandas DataFrame.')    
    if not isinstance(cols, (list,np.ndarray)):
        raise TypeError('cols is not array-like.')    
    if not isinstance(bins, (int,list,np.ndarray,dict)):
        raise TypeError('bins is of incorrect type.')
    if isinstance(bins, dict):
        for col in cols:
            if col not in bins:
                raise AssertionError('column {0} is not included in bins dictionary.'.format(col))

    # check for numeric bins
    for col in list(set(data._get_numeric_data().columns) - set(cols)):
        nuq = data[col].nunique()
        if (nuq > 0.9 * len(data)) or (nuq > 100):
            warnings.warn(
                "numeric variable {1:s} has {0:d} unique values. Are you sure you don't want to bin it?".format(nuq, str(col)), Warning)

    binned_data = data.copy()

    if isinstance(bins, (list, np.ndarray)):
        xbins = bins

    bins_dict = {}
    for col in cols:
        if isinstance(bins, (int, float)):
            xbins = bin_edges(data[col].astype(float), int(bins), quantile=quantile)
        if isinstance(bins, dict):
            if isinstance(bins[col], (int, float)):
                xbins = bin_edges(data[col].astype(float), int(bins[col]), quantile=quantile)
            elif isinstance(bins[col], (list, np.ndarray)):
                xbins = bins[col]
        binned_data[col], bin_labels = bin_array(data[col].astype(float).values, xbins)
        if retbins:
            bins_dict[col] = bin_labels

    if retbins:
        return binned_data, bins_dict

    return binned_data


def create_correlation_overview_table(vals:dict) -> pd.DataFrame:
    """
    Create overview table of phik/significance data.

    :param dict vals: dictionary holding data for each variable pair formatted as {'var1:var2' : value}
    :returns: symmetric table with phik/significances of all variable pairs
    :rtype: pandas.DataFrame
    """
    if not isinstance(vals, dict):
        raise TypeError('vals is not a dict.')    

    ll = []
    for k, v in vals.items():
        ll.append(k.split(':')+[v])
        ll.append(list(reversed(k.split(':')))+[v])

    corr_matrix = pd.DataFrame(ll, columns=['var1', 'var2', 'vals'])\
        .pivot_table(index='var1', columns='var2', values='vals')
        
    return corr_matrix


def hist2d_from_rebinned_df(data_binned:pd.DataFrame, dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True) -> pd.DataFrame:
    """
    Give binned 2d dataframe of two colums of rebinned input dataframe

    :param df: input data. Dataframe must contain exactly two columns
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :returns: histogram dataframe
    """
    if not isinstance(data_binned, pd.DataFrame):
        raise TypeError('data_binned is not a pandas DataFrame.')    

    assert len(data_binned.columns) == 2, 'DataFrame should contain only two columns'

    if not dropna:
        data_binned = data_binned.fillna(defs.NaN).copy()
    if drop_underflow:
        data_binned = data_binned.replace(defs.UF, np.nan).copy()
    if drop_overflow:
        data_binned = data_binned.replace(defs.OF, np.nan).copy()

    # create a contingency table
    c0, c1 = data_binned.columns
    df_datahist = data_binned.groupby([c0, c1])[c0].count().to_frame().unstack().fillna(0)
    df_datahist.columns = df_datahist.columns.droplevel()

    return df_datahist


def hist2d(df, interval_cols=None, bins=10, quantile:bool=False, dropna:bool=True, drop_underflow:bool=True,
           drop_overflow:bool=True, retbins:bool=False) -> pd.DataFrame:
    """
    Give binned 2d dataframe of two colums of input dataframe

    :param df: input data. Dataframe must contain exactly two columns
    :param interval_cols: columns with interval variables which need to be binned
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param bool quantile: when the number of bins is specified, use uniform binning (False) or quantile binning (True)
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :returns: histogram dataframe
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df is not a pandas DataFrame.')    
    if not isinstance(interval_cols, (type(None), list, np.ndarray)):
        raise TypeError('interval_cols is not None or a list.')    
    if not isinstance(bins, (int,float,list,np.ndarray,dict)):
        raise TypeError('bins is of incorrect type.')    

    assert len(df.columns) == 2, 'DataFrame should contain only two columns'

    if isinstance( interval_cols, type(None) ):
        interval_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if interval_cols:
            print('interval_cols not set, guessing: {0:s}'.format(str(interval_cols)))
    assert isinstance( interval_cols, list ), 'interval_cols is not a list.'

    data_binned, binning_dict = bin_data(df, interval_cols, retbins=True, bins=bins, quantile=quantile)
    datahist = hist2d_from_rebinned_df(data_binned, dropna=dropna, drop_underflow=drop_underflow, drop_overflow=drop_overflow)

    if retbins:
        return datahist, binning_dict

    return datahist
