"""Project: PhiK - correlation analyzer library

Created: 2018/09/05

Description:
    Functions for doing the significance evaluation of an hypothesis test of variable independence
    using a contingency table.

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""
from typing import Tuple, Union

import numpy as np
import pandas as pd
import math
import itertools
import warnings

from scipy import stats
from scipy import special, optimize

from phik import definitions as defs
from .binning import bin_data, create_correlation_overview_table
from .statistics import get_chi2_using_dependent_frequency_estimates
from .statistics import estimate_ndof, theoretical_ndof
from .simulation import sim_chi2_distribution
from .data_quality import dq_check_nunique_values, dq_check_hist2d
from .utils import array_like_to_dataframe, guess_interval_cols


def fit_test_statistic_distribution(
    chi2s: Union[list, np.ndarray], nbins: int = 50
) -> Tuple[float, float, float, float]:
    """
    Fit the hybrid chi2-distribution to the data to find f.

    Perform a binned likelihood fit to the data to find the optimal value for the fraction f in
    h(x|f) = N * (f * chi2(x, ndof) + (1-f) * gauss(x, ndof, sqrt(ndof))
    The parameter ndof is fixed in the fit using ndof = mean(x). The total number of datapoints N is also fixed.

    :param list chi2s: input data - a list of chi2 values
    :param int nbins: in order to fit the data a histogram is created with nbins number of bins
    :returns: f, ndof, sigma (width of gauss), bw (bin width)
    """

    def myfunc(x, N, f, k, sigma):
        return N * (f * stats.chi2.pdf(x, k) + (1 - f) * stats.norm.pdf(x, k, sigma))

    ffunc = lambda x, f: myfunc(x, nsim * bw, f, kmean, lsigma)

    def gtest(p, x, y):
        f = ffunc(x, *p)
        ll = f - special.xlogy(y, f) + special.gammaln(y + 1)
        return np.sqrt(ll)

    kmean = np.mean(chi2s)  # NOTE: this is used to fix kmean in the fit!
    lsigma = np.sqrt(kmean)  # NOTE: this is used to fix sigma in the fit!
    nsim = len(chi2s)  # NOTE: this is used to fix N in fit (N=nsim*bw) !

    yhist, xbounds = np.histogram(chi2s, bins=nbins)
    bw = xbounds[1] - xbounds[0]  # NOTE: this is used to fix N in fit (N=nsim*bw) !
    xhist = xbounds[:-1] + np.diff(xbounds) / 2

    initGuess = (1.0,)  # starting value for parameter f
    res = optimize.least_squares(
        gtest, initGuess, bounds=((0.0,), (1,)), args=(xhist, yhist)
    )

    return res.x[0], kmean, lsigma, bw


def hfunc(x: float, N: float, f: float, k: float, sigma: float) -> float:
    """
    Definition of the combined probability density function h(x)

    h(x|f) = N * (f * chi2(x, k) + (1-f) * gauss(x, k, sigma)

    :param float x: x
    :param float N: normalisation
    :param float f: fraction [0,1]
    :param float k: ndof of chi2 function and mean of gauss
    :param float sigma: width of gauss
    :return: h(x|f)
    """
    return N * (f * stats.chi2.pdf(x, k) + (1 - f) * stats.norm.pdf(x, k, sigma))


def significance_from_chi2_ndof(chi2: float, ndof: float) -> Tuple[float, float]:
    """
    Convert a chi2 into significance using knowledge about the number of degrees of freedom

    Conversion is done using asymptotic approximation.

    :param float chi2: chi2 value
    :param float ndof: number of degrees of freedom
    :returns: p_value, significance
    """
    p_value = stats.chi2.sf(chi2, ndof)
    z_value = -stats.norm.ppf(p_value)

    # scenario where p_value is too small to evaluate Z
    # use Chernoff approximation for p-value upper bound
    # see: https://en.wikipedia.org/wiki/Chi-squared_distribution
    if p_value == 0:
        z = chi2 / ndof
        u = -math.log(2 * math.pi) - ndof * math.log(z) + ndof * (z - 1)
        z_value = math.sqrt(u - math.log(u))

    return p_value, z_value


def significance_from_chi2_asymptotic(
    values: np.ndarray, chi2: float
) -> Tuple[float, float]:
    """
    Convert a chi2 into significance using knowledge about the number of degrees of freedom

    Convention is done using asymptotic approximation.

    :param float chi2: chi2 value
    :param float ndof: number of degrees of freedom
    :returns: p_value, significance
    """

    ndof = theoretical_ndof(values)
    p_value, z_value = significance_from_chi2_ndof(chi2, ndof)

    return p_value, z_value


def significance_from_chi2_MC(
    chi2: float,
    values: np.ndarray,
    nsim: int = 1000,
    lambda_: str = "log-likelihood",
    simulation_method: str = "multinominal",
    chi2s=None,
    njobs: int = -1,
) -> Tuple[float, float]:
    """
    Convert a chi2 into significance using knowledge about the shape of the chi2 distribution of simulated data

    Calculate significance based on simulation (MC method), using a simple percentile.

    :param float chi2: chi2 value
    :param list chi2s: provide your own chi2s values (optional)
    :param int njobs: number of parallel jobs used for simulation. default is -1. 1 uses no parallel jobs.
    :returns: pvalue, significance
    """

    # determine effective number of degrees of freedom using simulation
    if chi2s is None:
        chi2s = sim_chi2_distribution(
            values,
            nsim=nsim,
            lambda_=lambda_,
            simulation_method=simulation_method,
            njobs=njobs,
        )

    # calculate p_value based on simulation (MC method)
    empirical_p_value = 1.0 - stats.percentileofscore(chi2s, chi2) / 100.0
    empirical_z_value = -stats.norm.ppf(empirical_p_value)

    return empirical_p_value, empirical_z_value


def significance_from_chi2_hybrid(
    chi2: float,
    values: np.ndarray,
    nsim: int = 1000,
    lambda_: str = "log-likelihood",
    simulation_method: str = "multinominal",
    chi2s=None,
    njobs: int = -1,
) -> Tuple[float, float]:
    """
    Convert a chi2 into significance using a hybrid method

    This method combines the asymptotic method with the MC method, but applies several corrections:

    * use effective number of degrees of freedom instead of number of degrees of freedom. The effective number of\
      degrees of freedom is measured as mean(chi2s), with chi2s a list of simulated chi2 values.
    * for low statistics data sets, with on average less than 4 data points per bin, the distribution of chi2-values\
      is better described by h(x|f) then by the usual chi2-distribution. Use h(x|f) to convert the chi2 value to \
      the pvalue and significance.

    h(x|f) = N * (f * chi2(x, ndof) + (1-f) * gauss(x, ndof, sqrt(ndof))

    :param float chi2: chi2 value
    :param list chi2s: provide your own chi2s values (optional)
    :param float avg_per_bin: average number of data points per bin
    :param int njobs: number of parallel jobs used for simulation. default is -1. 1 uses no parallel jobs.
    :returns: p_value, significance
    """

    # determine effective number of degrees of freedom using simulation
    if chi2s is None:
        chi2s = sim_chi2_distribution(
            values,
            nsim=nsim,
            lambda_=lambda_,
            simulation_method=simulation_method,
            njobs=njobs,
        )

    # average number of records per bin
    avg_per_bin = values.sum() / values.shape[0] * values.shape[1]

    if avg_per_bin <= 4:
        f, endof, lsigma, bw = fit_test_statistic_distribution(chi2s)
        pvalue_h = f * stats.chi2.sf(chi2, endof) + (1 - f) * stats.norm.sf(
            chi2, endof, lsigma
        )
    else:
        endof = estimate_ndof(chi2s)
        pvalue_h = stats.chi2.sf(chi2, endof)

    zvalue_h = -stats.norm.ppf(pvalue_h)

    if pvalue_h == 0:
        # apply Chernoff approximation as upper bound for p-value
        # see: https://en.wikipedia.org/wiki/Chi-squared_distribution
        z = chi2 / endof
        u = -math.log(2 * math.pi) - endof * math.log(z) + endof * (z - 1)
        if avg_per_bin <= 4:
            u += -2 * math.log(f)
        zvalue_h = math.sqrt(u - math.log(u))

    return pvalue_h, zvalue_h


def significance_from_hist2d(
    values: np.ndarray,
    nsim: int = 1000,
    lambda_: str = "log-likelihood",
    simulation_method: str = "multinominal",
    significance_method: str = "hybrid",
    njobs: int = -1,
) -> Tuple[float, float]:
    """
    Calculate the significance of correlation of two variables based on the contingency table

    :param values: contingency table
    :param int nsim: number of simulations
    :param str lambda_: test statistic. Available options are [pearson, log-likelihood]
    :param str simulation_method: simulation method. Options: [multinominal, row_product_multinominal, \
     col_product_multinominal, hypergeometric].
    :param str significance_method: significance_method. Options: [asymptotic, MC, hybrid]
    :param int njobs: number of parallel jobs used for simulation. default is -1. 1 uses no parallel jobs.
    :return: pvalue, significance
    """

    # chi2 of the data
    chi2 = get_chi2_using_dependent_frequency_estimates(values, lambda_=lambda_)

    if significance_method == "asymptotic":
        # calculate pvalue and zvalue based on chi2 and ndof (asymptotic method)
        pvalue, zvalue = significance_from_chi2_asymptotic(values, chi2)
    elif significance_method == "MC":
        # calculate pvalue based on simulation (MC method)
        pvalue, zvalue = significance_from_chi2_MC(
            chi2,
            values,
            nsim=nsim,
            lambda_=lambda_,
            simulation_method=simulation_method,
            njobs=njobs,
        )
    elif significance_method == "hybrid":
        # low statistics : calculate pvalue and zvalue using h(x|f) and endof
        # high statistics: calculate pvalue and zvalue using chi2-distribution and endof
        pvalue, zvalue = significance_from_chi2_hybrid(
            chi2,
            values,
            nsim=nsim,
            lambda_=lambda_,
            simulation_method=simulation_method,
            njobs=njobs,
        )
    else:
        raise NotImplementedError(
            "simulation_method {0:s} is unknown".format(simulation_method)
        )

    return pvalue, zvalue


def significance_from_rebinned_df(
    data_binned: pd.DataFrame,
    lambda_: str = "log-likelihood",
    simulation_method: str = "multinominal",
    nsim: int = 1000,
    significance_method: str = "hybrid",
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
    njobs: int = -1,
) -> pd.DataFrame:
    """
    Calculate significance of correlation of all variable combinations in the DataFrame

    :param data_binned: input binned DataFrame
    :param int nsim: number of simulations
    :param str lambda_: test statistic. Available options are [pearson, log-likelihood]
    :param str simulation_method: simulation method. Options: [mutlinominal, row_product_multinominal, \
     col_product_multinominal, hypergeometric].
    :param str significance_method: significance_method. Options: [asymptotic, MC, hybrid]
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :param int njobs: number of parallel jobs used for simulation. default is -1.
    :return: significance matrix
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
    signifs = []
    for i, (c0, c1) in enumerate(
        itertools.combinations_with_replacement(data_binned.columns.values, 2)
    ):
        datahist = (
            data_binned.groupby([c0, c1])[c0].count().to_frame().unstack().fillna(0)
        )
        if 1 in datahist.shape or 0 in datahist.shape:
            signifs.append((c0, c1, np.nan))
            warnings.warn(
                "Too few unique values for variable {0:s} ({1:d}) or {2:s} ({3:d}) to calculate significance".format(
                    c0, datahist.shape[0], c1, datahist.shape[1]
                )
            )
            continue

        datahist.columns = datahist.columns.droplevel()
        datahist = datahist.values
        pvalue, zvalue = significance_from_hist2d(
            datahist,
            nsim=nsim,
            lambda_=lambda_,
            simulation_method=simulation_method,
            significance_method=significance_method,
            njobs=njobs,
        )
        signifs.append((c0, c1, zvalue))

    if len(signifs) == 0:
        return pd.DataFrame(np.nan, index=column_order, columns=column_order)

    significance_overview = create_correlation_overview_table(signifs)

    # restore column order
    significance_overview = significance_overview.reindex(columns=column_order)
    significance_overview = significance_overview.reindex(index=column_order)

    return significance_overview


def significance_matrix(
    df: pd.DataFrame,
    interval_cols: list = None,
    lambda_: str = "log-likelihood",
    simulation_method: str = "multinominal",
    nsim: int = 1000,
    significance_method: str = "hybrid",
    bins: Union[int, list, np.ndarray, dict] = 10,
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
    verbose: bool = True,
    njobs: int = -1,
) -> pd.DataFrame:
    """
    Calculate significance of correlation of all variable combinations in the dataframe

    :param pd.DataFrame df: input data
    :param list interval_cols: column names of columns with interval variables.
    :param int nsim: number of simulations
    :param str lambda_: test statistic. Available options are [pearson, log-likelihood]
    :param str simulation_method: simulation method. Options: [mutlinominal, row_product_multinominal, \
     col_product_multinominal, hypergeometric].
    :param int nsim: number of simulated datasets
    :param str significance_method: significance_method. Options: [asymptotic, MC, hybrid]
        :param bool dropna: remove NaN values with True
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :param bool verbose: if False, do not print all interval columns that are guessed
    :param int njobs: number of parallel jobs used for simulation. default is -1. 1 uses no parallel jobs.
    :return: significance matrix
    """

    if interval_cols is None:
        interval_cols = guess_interval_cols(df, verbose)

    df_clean, interval_cols_clean = dq_check_nunique_values(
        df, interval_cols, dropna=dropna
    )

    data_binned = bin_data(df_clean, interval_cols_clean, bins=bins)

    return significance_from_rebinned_df(
        data_binned,
        lambda_=lambda_,
        simulation_method=simulation_method,
        nsim=nsim,
        significance_method=significance_method,
        dropna=dropna,
        drop_underflow=drop_underflow,
        drop_overflow=drop_overflow,
        njobs=njobs,
    )


def significance_from_array(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    num_vars=None,
    bins: Union[int, list, np.ndarray, dict] = 10,
    quantile: bool = False,
    lambda_: str = "log-likelihood",
    nsim: int = 1000,
    significance_method: str = "hybrid",
    simulation_method: str = "multinominal",
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
    njobs: int = -1,
) -> Tuple[float, float]:
    """
    Calculate the significance of correlation

    Calculate the significance of correlation for two variables which can be of interval, oridnal or categorical type.\
    Interval variables will be binned.

    :param x: array-like input
    :param y: array-like input
    :param num_vars: list of numeric variables which need to be binned, e.g. ['x'] or ['x','y']
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param quantile: when bins is an integer, uniform bins (False) or bins based on quantiles (True)
    :param str lambda_: test statistic. Available options are [pearson, log-likelihood]
    :param int nsim: number of simulated datasets
    :param str simulation_method: simulation method. Options: [mutlinominal, row_product_multinominal, \
    col_product_multinominal, hypergeometric].
    :param str significance_method: significance_method. Options: [asymptotic, MC, hybrid]
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :param int njobs: number of parallel jobs used for simulation. default is -1. 1 uses no parallel jobs.
    :return: p-value, significance
    """
    if num_vars is None:
        num_vars = []
    elif isinstance(num_vars, str):
        num_vars = [num_vars]

    if len(num_vars) > 0:
        df = array_like_to_dataframe(x, y)
        x, y = bin_data(df, num_vars, bins=bins, quantile=quantile).T.values

    return significance_from_binned_array(
        x,
        y,
        lambda_=lambda_,
        significance_method=significance_method,
        nsim=nsim,
        simulation_method=simulation_method,
        dropna=dropna,
        drop_underflow=drop_underflow,
        drop_overflow=drop_overflow,
        njobs=njobs,
    )


def significance_from_binned_array(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    lambda_: str = "log-likelihood",
    significance_method: str = "hybrid",
    nsim: int = 1000,
    simulation_method: str = "multinominal",
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
    njobs: int = -1,
) -> Tuple[float, float]:
    """
    Calculate the significance of correlation

    Calculate the significance of correlation for two variables which can be of interval, oridnal or categorical type. \
    Interval variables need to be binned.

    :param x: array-like input
    :param y: array-like input
    :param str lambda_: test statistic. Available options are [pearson, log-likelihood]
    :param str simulation_method: simulation method. Options: [multinominal, row_product_multinominal, \
    col_product_multinominal, hypergeometric].
    :param int nsim: number of simulated datasets
    :param str significance_method: significance_method. Options: [asymptotic, MC, hybrid]
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :param int njobs: number of parallel jobs used for simulation. default is -1. 1 uses no parallel jobs.
    :return: p-value, significance
    """

    if not dropna:
        x = pd.Series(x).fillna(defs.NaN).astype(str).values
        y = (
            pd.Series(y).fillna(defs.NaN).astype(str).values
        )  # crosstab cannot handle mixed type y!

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
        return np.nan, np.nan

    return significance_from_hist2d(
        hist2d,
        lambda_=lambda_,
        significance_method=significance_method,
        simulation_method=simulation_method,
        nsim=nsim,
        njobs=njobs,
    )
