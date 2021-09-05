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
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd

from phik import definitions as defs
from phik.utils import array_like_to_dataframe, guess_interval_cols
from phik.data_quality import dq_check_nunique_values


def bin_edges(
    arr: Union[np.ndarray, list, pd.Series], nbins: int, quantile: bool = False
) -> np.ndarray:
    """
    Create uniform or quantile bin-edges for the input array.

    :param arr: array like object with input data
    :param int nbins: the number of bin
    :param bool quantile: uniform bins (False) or bins based on quantiles (True)
    :returns: array with bin edges
    """

    if quantile:
        quantiles = np.linspace(0, 1, nbins + 1)
        xbins = np.quantile(arr[~np.isnan(arr)], quantiles)
        xbins[0] -= 1e-14
    else:
        xbins = np.linspace(
            np.min(arr[~np.isnan(arr)]) - 1e-14, np.max(arr[~np.isnan(arr)]), nbins + 1
        )

    return xbins


def bin_array(
    arr: Union[np.ndarray, list], bin_edges: Union[np.ndarray, list]
) -> Tuple[np.ndarray, list]:
    """
    Index the data given the bin_edges.

    Underflow and overflow values are indicated.

    :param arr: array like object with input data
    :param bin_edges: list with bin edges.
    :returns: indexed data
    """

    # Bin data
    binned_arr = np.searchsorted(bin_edges, arr).astype(object)

    # Check if all bins are filled and store bin-labels
    bin_labels = []
    bin_indices = pd.Series(binned_arr).value_counts().index
    for i in range(1, len(bin_edges)):
        if i in bin_indices:
            bin_labels.append((bin_edges[i - 1], bin_edges[i]))

    # NaN values are added to the overflow bin. Restore NaN values:
    binned_arr[np.argwhere(np.isnan(arr))] = np.nan

    # Set underflow values to UF
    binned_arr[np.argwhere(binned_arr == 0)] = defs.UF

    # Set overflow values to OF
    binned_arr[np.argwhere(binned_arr == len(bin_edges))] = defs.OF

    return binned_arr, bin_labels


def bin_data(
    data: pd.DataFrame,
    cols: Union[list, np.ndarray, tuple] = (),
    bins: Union[int, list, np.ndarray, dict] = 10,
    quantile: bool = False,
    retbins: bool = False,
):
    """
    Index the input DataFrame given the bin_edges for the columns specified in cols.

    :param DataFrame data: input data
    :param list cols: list of columns with numeric data which needs to be indexed
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param quantile: when bins is an integer, uniform bins (False) or bins based on quantiles (True)
    :returns: rebinned DataFrame
    :rtype: pandas.DataFrame
    """
    xbins = None
    if isinstance(bins, dict):
        for col in cols:
            if col not in bins:
                raise ValueError(
                    "column {0} is not included in bins dictionary.".format(col)
                )
    elif isinstance(bins, (list, np.ndarray)):
        xbins = bins

    # MB 20210307: check for numeric bins turned off here, also done in dq_check_nunique_values().

    binned_data = data.copy()

    bins_dict = {}
    for col in cols:
        if np.issubdtype(type(bins), np.integer) or np.issubdtype(
            type(bins), np.floating
        ):
            xbins = bin_edges(data[col].astype(float), int(bins), quantile=quantile)
        elif isinstance(bins, dict):
            if np.issubdtype(type(bins[col]), np.integer) or np.issubdtype(
                type(bins[col]), np.floating
            ):
                xbins = bin_edges(
                    data[col].astype(float), int(bins[col]), quantile=quantile
                )
            elif isinstance(bins[col], (list, np.ndarray)):
                xbins = bins[col]
        elif xbins is None:
            raise ValueError(
                "Unexpected type for bins. The found type was '%s'" % str(type(bins))
            )

        binned_data[col], bin_labels = bin_array(data[col].astype(float).values, xbins)
        if retbins:
            bins_dict[col] = bin_labels

    if retbins:
        return binned_data, bins_dict

    return binned_data


def auto_bin_data(
    df: pd.DataFrame,
    interval_cols: Optional[list] = None,
    bins: Union[int, list, np.ndarray, dict] = 10,
    quantile: bool = False,
    dropna: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Index the input DataFrame with automatic bin_edges and interval columns.

    :param pd.DataFrame data_binned: input data
    :param list interval_cols: column names of columns with interval variables.
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column
        the bins are specified. (default=10)\
        E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param quantile: when bins is an integer, uniform bins (False) or bins based on quantiles (True)
    :param bool dropna: remove NaN values with True
    :param bool verbose: if False, do not print all interval columns that are guessed
    :return: phik correlation matrix
    """
    # guess interval columns
    if interval_cols is None:
        interval_cols = guess_interval_cols(df, verbose)

    # clean the data
    df_clean, interval_cols_clean = dq_check_nunique_values(
        df, interval_cols, dropna=dropna
    )

    # perform rebinning
    data_binned, binning_dict = bin_data(
        df_clean, cols=interval_cols_clean, bins=bins, quantile=quantile, retbins=True
    )
    return data_binned, binning_dict


def create_correlation_overview_table(
    vals: List[Tuple[str, str, float]]
) -> pd.DataFrame:
    """
    Create overview table of phik/significance data.

    :param list vals: list holding tuples of data for each variable pair formatted as ('var1', 'var2', value)
    :returns: symmetric table with phik/significances of all variable pairs
    :rtype: pandas.DataFrame
    """

    ll = []
    for c0, c1, v in vals:
        ll.append([c0, c1, v])
        ll.append([c1, c0, v])

    corr_matrix = pd.DataFrame(ll, columns=["var1", "var2", "vals"]).pivot_table(
        index="var1", columns="var2", values="vals"
    )
    corr_matrix.columns.name = None
    corr_matrix.index.name = None
    return corr_matrix


def hist2d_from_rebinned_df(
    data_binned: pd.DataFrame,
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
) -> pd.DataFrame:
    """
    Give binned 2d DataFrame of two columns of rebinned input DataFrame

    :param df: input data. DataFrame must contain exactly two columns
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :returns: histogram DataFrame
    """

    c0, c1 = data_binned.columns

    if not dropna:
        data_binned.fillna(defs.NaN, inplace=True)
    if drop_underflow:
        data_binned.replace(defs.UF, np.nan, inplace=True)
    if drop_overflow:
        data_binned.replace(defs.OF, np.nan, inplace=True)

    # create a contingency table
    df_datahist = (
        data_binned.groupby([c0, c1])[c0].count().to_frame().unstack().fillna(0)
    )
    df_datahist.columns = df_datahist.columns.droplevel()

    return df_datahist


def hist2d(
    df: pd.DataFrame,
    interval_cols: Optional[Union[list, np.ndarray]] = None,
    bins: Union[int, float, list, np.ndarray, dict] = 10,
    quantile: bool = False,
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
    retbins: bool = False,
    verbose: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Give binned 2d DataFrame of two columns of input DataFrame

    :param df: input data. DataFrame must contain exactly two columns
    :param interval_cols: columns with interval variables which need to be binned
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param bool quantile: when the number of bins is specified, use uniform binning (False) or quantile binning (True)
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :param bool verbose: if False, do not print all interval columns that are guessed
    :returns: histogram DataFrame
    """

    if len(df.columns) != 2:
        raise ValueError("DataFrame should contain only two columns")

    if interval_cols is None:
        interval_cols = guess_interval_cols(df, verbose)

    data_binned, binning_dict = bin_data(
        df, interval_cols, retbins=True, bins=bins, quantile=quantile
    )
    datahist = hist2d_from_rebinned_df(
        data_binned,
        dropna=dropna,
        drop_underflow=drop_underflow,
        drop_overflow=drop_overflow,
    )

    if retbins:
        return datahist, binning_dict

    return datahist


def hist2d_from_array(
    x: Union[pd.Series, list, np.ndarray], y: [pd.Series, list, np.ndarray], **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Give binned 2d DataFrame of two input arrays

    :param x: input data. First array-like.
    :param y: input data. Second array-like.
    :param interval_cols: columns with interval variables which need to be binned
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param bool quantile: when the number of bins is specified, use uniform binning (False) or quantile binning (True)
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :returns: histogram DataFrame
    """

    df = array_like_to_dataframe(x, y)
    return hist2d(df, **kwargs)
