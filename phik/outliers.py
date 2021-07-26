"""Project: PhiK - correlation analyzer library

Created: 2018/09/05

Description:
    Functions for calculating the statistical significance of outliers in a contingency table.

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

from typing import Tuple, Union, Optional
import itertools
import numpy as np
import pandas as pd
import warnings

from scipy import stats
from scipy.special import betainc

from phik import definitions as defs
from .binning import bin_data, hist2d_from_rebinned_df
from .betainc import log_incompbeta
from .statistics import z_from_logp
from .data_quality import dq_check_nunique_values
from .utils import array_like_to_dataframe, guess_interval_cols


def poisson_obs_p(nobs: int, nexp: float, nexperr: float) -> float:
    """
    Calculate p-value for nobs observations given the expected value and its
    uncertainty using the Linnemann method.

    If the uncertainty
    on the expected value is known the Linnemann method is used. Otherwise the Poisson distribution is
    used to estimate the p-value.

    Measures of Significance in HEP and Astrophysics
    Authors: J. T. Linnemann
    http://arxiv.org/abs/physics/0312059

    Code inspired by:
    https://root.cern.ch/doc/master/NumberCountingUtils_8cxx_source.html#l00086

    Three fixes are added for:

      * nobs = 0, when - by construction - p should be 1.
      * uncertainty of zero, for which Linnemann's function does not work, but one can simply revert to regular Poisson.
      * when nexp=0, betainc always returns 1. Here we set nexp = nexperr.

    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: p-value
    :rtype: float
    """
    if nobs == 0:
        return 1

    if nexperr > 0:
        nexpalt = nexp if nexp > 0 else nexperr
        tau = nexpalt / (nexperr * nexperr)
        b = nexpalt * tau + 1
        x = 1 / (1 + tau)
        p = betainc(nobs, b, x)
    else:  # assume error==0
        p = stats.poisson.sf(nobs - 1, nexp)

    return p


def log_poisson_obs_p(nobs: int, nexp: float, nexperr: float) -> Tuple[float, float]:
    """
    Calculate logarithm of p-value for nobs observations given the expected value and its
    uncertainty using the Linnemann method.

    If the uncertainty
    on the expected value is known the Linnemann method is used. Otherwise the Poisson distribution is
    used to estimate the p-value.

    Measures of Significance in HEP and Astrophysics
    Authors: J. T. Linnemann
    http://arxiv.org/abs/physics/0312059

    Code inspired by:
    https://root.cern.ch/doc/master/NumberCountingUtils_8cxx_source.html#l00086

    Three fixes are added for:

      * nobs = 0, when - by construction - p should be 1.
      * uncertainty of zero, for which Linnemann's function does not work, but one can simply revert to regular Poisson.
      * when nexp=0, betainc always returns 1. Here we set nexp = nexperr.

    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: tuple containing pvalue and 1 - pvalue
    :rtype: tuple
    """
    if nobs == 0:
        # p=1, 1-p=0 --> logp=0,log(1-p)=-inf
        return 0, -np.inf

    if nexperr > 0:
        nexpalt = nexp if nexp > 0 else nexperr
        tau = nexpalt / (nexperr * nexperr)
        b = nexpalt * tau + 1
        x = 1 / (1 + tau)
        tlogp = log_incompbeta(nobs, b, x)
    else:  # assume error==0. nobs>0 at this stage
        logp = stats.poisson.logsf(nobs - 1, nexp)
        p = stats.poisson.sf(nobs - 1, nexp)
        tlogp = (logp, np.log(1 - p))

    return tlogp


def poisson_obs_z(nobs: int, nexp: float, nexperr: float) -> float:
    """
    Calculate the Z-value for measuring nobs observations given the expected value.

    The Z-value express the number
    of sigmas the observed value deviates from the expected value, and is based on the p-value calculation.
    If the uncertainty on the expected value is known the Linnemann method is used. Otherwise the Poisson distribution is used to estimate the p-value.

    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: Z-value
    :rtype: float
    """
    p_value = poisson_obs_p(nobs, nexp, nexperr)

    # special cases: numerically too close to zero or one.
    # try to evaluate log(p) or log(1-p)
    if p_value == 0 or p_value == 1:
        tlogp = log_poisson_obs_p(nobs, nexp, nexperr)
        if p_value == 0:
            logp = tlogp[0]
            z_value = z_from_logp(logp)
        else:
            log1mp = tlogp[1]
            z_value = z_from_logp(log1mp, flip_sign=True)
    # default:
    else:
        z_value = -stats.norm.ppf(p_value)

    return z_value


def poisson_obs_mid_p(nobs: int, nexp: float, nexperr: float) -> float:
    """
    Calculate the p-value for measuring nobs observations given the expected value.

    The Lancaster mid-P correction is applied to take into account the effects of discrete statistics.
    If the uncertainty on the expected value is known the Linnemann method is used for the p-value calculation.
    Otherwise the Poisson distribution is used to estimate the p-value.

    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: mid p-value
    :rtype: float
    """
    p = poisson_obs_p(nobs, nexp, nexperr)
    pplus1 = poisson_obs_p(nobs + 1, nexp, nexperr)
    mid_p = 0.5 * (p - pplus1)
    p -= mid_p

    return p


def log_poisson_obs_mid_p(
    nobs: int, nexp: float, nexperr: float
) -> Tuple[float, float]:
    """
    Calculate the logarithm of the p-value for measuring nobs observations given the expected value.

    The Lancaster mid-P correction is
    applied to take into account the effects of discrete statistics. If the uncertainty on the expected value is known the
    Linnemann method is used for the p-value calculation. Otherwise the Poisson distribution is used to estimate the p-value.

    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: tuple of log(p) and log(1-p)
    :rtype: tuple
    """
    tlogp = log_poisson_obs_p(nobs, nexp, nexperr)
    tlogpp1 = log_poisson_obs_p(nobs + 1, nexp, nexperr)

    # 1. evaluate log([p+pp1]/2) ; note that p > pp1
    #    = log(0.5) + log(p) + log(1 + exp[log(pp1)-log(p)])
    lp = tlogp[0]
    lp1 = tlogpp1[0]
    logmidp = np.log(0.5) + lp + np.log(1 + np.exp(lp1 - lp))

    # 2. let q = 1 - p; note that qp1 > q
    #    evaluate log(1-[p+pp1]/2) = log ([q+qp1]/2)
    #    = log(0.5) + log(qp1) + log(1 + exp[log(q)-log(qp1)])
    lq = tlogp[1]
    lq1 = tlogpp1[1]
    logmidq = np.log(0.5) + lq1 + np.log(1 + np.exp(lq - lq1))

    return logmidp, logmidq


def poisson_obs_mid_z(nobs: int, nexp: float, nexperr: float) -> float:
    """Calculate the Z-value for measuring nobs observations given the expected value.

    The Z-value express the number
    of sigmas the observed value deviates from the expected value, and is based on the p-value calculation.
    The Lancaster midP correction is applied to take into account the effects of low statistics. If the uncertainty on the
    expected value is known the Linnemann method is used for the p-value calculation. Otherwise the Poisson distribution is
    used to estimate the p-value.

    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: Z-value
    :rtype: tuple
    """
    p_value = poisson_obs_mid_p(nobs, nexp, nexperr)

    # special cases: numerically too close to zero or one.
    # try to evaluate log(p) or log(1-p)
    if p_value == 0 or p_value == 1:
        tlogp = log_poisson_obs_mid_p(nobs, nexp, nexperr)
        if p_value == 0:
            logp = tlogp[0]
            z_value = z_from_logp(logp)
        else:
            log1mp = tlogp[1]
            z_value = z_from_logp(log1mp, flip_sign=True)
    # default:
    else:
        z_value = -stats.norm.ppf(p_value)

    return z_value


def get_independent_frequency_estimates(
    values: np.ndarray, CI_method: str = "poisson"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculation of expected frequencies, based on the ABCD-method, i.e. independent frequency estimates.

    :param values: The contingency table. The table contains the observed number of occurrences in each category.
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error.\
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :returns exp, experr: expected frequencies, error on the expected frequencies
    """

    # Initialize
    exp = np.zeros(values.shape)
    experr = np.zeros(values.shape)

    # Calculate dependent expected value using ABCD method
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            Aobs = values[i][j]
            B = values[i].sum() - Aobs
            C = values[:, j].sum() - Aobs
            D = values.sum() - B - C - Aobs
            # prediction for A can only be calculated if D is non-zero
            if D > 0:
                exp[i][j] = B * C / D
                sigmaB = get_uncertainty(B, CI_method=CI_method)
                sigmaC = get_uncertainty(C, CI_method=CI_method)
                sigmaD = get_uncertainty(D, CI_method=CI_method)
                experr[i][j] = np.sqrt(
                    pow(sigmaB * C / D, 2)
                    + pow(sigmaC * B / D, 2)
                    + pow(sigmaD * exp[i][j] / D, 2)
                )
            # in case of zero D, A is infinity. Set prediction to NaN.
            else:
                exp[i][j] = np.nan
                experr[i][j] = np.nan

    return exp, experr


def get_uncertainty(x: float, CI_method: str = "poisson") -> float:
    """
    Calculate the uncertainty on a random number x taken from the poisson distribution

    The uncertainty on the x is calculated using either the standard poisson error (poisson) or using the asymmetric
    exact poisson interval (exact_poisson).
    https://www.ncbi.nlm.nih.gov/pubmed/2296988 #FIXME: check ref

    :param float x: value, must be equal or greater than zero
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error.\
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :return x_err: the uncertainty on x (1 sigma)
    """

    if CI_method == "exact_poisson":
        xerr = get_exact_poisson_uncertainty(x)
    elif CI_method == "poisson":
        xerr = get_poisson_uncertainty(x)
    else:
        raise NotImplementedError("CI method {} not valid".format(CI_method))

    return xerr


def get_poisson_uncertainty(x: float) -> float:
    """
    Calculate the uncertainty on x using standard poisson error. In case x=0 the error=1 is assigned.

    :param float x: value
    :return x_err: the uncertainty on x (1 sigma)
    :rtype: float
    """
    return np.sqrt(x) if x >= 1 else 1.0


def get_exact_poisson_uncertainty(x: float, nsigmas: float = 1) -> float:
    """
    Calculate the uncertainty on x using an exact poisson confidence interval. The width of the confidence interval can
    be specified using the number of sigmas. The default number of sigmas is set to 1, resulting in an error that is
    approximated by the standard poisson error sqrt(x).

    Exact poisson uncertainty is described here:
    https://ms.mcmaster.ca/peter/s743/poissonalpha.html
    https://www.statsdirect.com/help/rates/poisson_rate_ci.htm
    https://www.ncbi.nlm.nih.gov/pubmed/2296988

    :param float x: value
    :return x_err: the uncertainty on x (1 sigma)
    :rtype: float
    """
    # see formula at:
    # https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
    pl = stats.norm.cdf(-1 * nsigmas, loc=0, scale=1)
    pu = stats.norm.cdf(1 * nsigmas, loc=0, scale=1)

    lb = stats.chi2.ppf(pl, 2 * x) / 2 if x != 0 else 0
    ub = stats.chi2.ppf(pu, 2 * (x + 1)) / 2

    # average err is almost equal to sqrt(x)+0.5
    return (ub - lb) / 2


def get_outlier_significances(
    obs: np.ndarray, exp: np.ndarray, experr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluation of significance of observation

    Evaluation of the significance of the difference between the observed number of occurrences and the expected number of
    occurrences, taking into account the uncertainty on the expected number of occurrences. When the uncertainty is
    not zero, the Linnemann method is used to calculate the p-values.

    :param obs: observed numbers
    :param exp: expected numbers
    :param experr: uncertainty on the expected numbers
    :returns: pvalues, zvalues
    """

    pvalues = np.zeros(obs.shape)
    zvalues = np.zeros(obs.shape)
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            pvalues[i][j] = poisson_obs_mid_p(obs[i][j], exp[i][j], experr[i][j])
            zvalues[i][j] = poisson_obs_mid_z(obs[i][j], exp[i][j], experr[i][j])

    return pvalues, zvalues


def outlier_significance_matrix_from_hist2d(
    data: np.ndarray, CI_method: str = "poisson"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the significance matrix of excesses or deficits in a contingency table

    :param data: numpy array contingency table
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error.\
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :return: p-value matrix, outlier significance matrix
    """

    # get expected values
    exp, experr = get_independent_frequency_estimates(data, CI_method=CI_method)
    pvalues, zvalues = get_outlier_significances(data, exp, experr)

    return pvalues, zvalues


def outlier_significance_matrix_from_rebinned_df(
    data_binned: pd.DataFrame,
    binning_dict: dict,
    CI_method: str = "poisson",
    ndecimals: int = 1,
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
) -> pd.DataFrame:
    """
    Calculate the significance matrix of excesses or deficits

    :param data_binned: input data. DataFrame must contain exactly two columns
    :param dict binning_dict: dictionary with bin edges for each binned interval variable. When no bin_edges are\
    provided values are used as bin label. Otherwise, bin labels are constructed based on provided bin edge information.
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error. \
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param ndecimals: number of decimals to use in labels of binned interval variables to specify bin edges (default=1)
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :return: outlier significance matrix (pd.DataFrame)
    """

    c0, c1 = data_binned.columns
    df_datahist = hist2d_from_rebinned_df(
        data_binned, dropna, drop_underflow, drop_overflow
    )

    if 1 in df_datahist.shape or 0 in df_datahist.shape:
        warnings.warn(
            "Too few unique values for variable {0:s} ({1:d}) or {2:s} ({3:d}) to calculate outlier "
            "significances".format(c0, df_datahist.shape[0], c1, df_datahist.shape[1])
        )
        return np.nan

    for c, a in [(c0, "index"), (c1, "columns")]:
        if c in binning_dict.keys():
            # check for missing bins. This can occur due to NaN values for variable c1 in which case rows are dropped
            orig_vals = (
                data_binned[~data_binned[c].isin([defs.UF, defs.OF, defs.NaN])][c]
                .value_counts()
                .sort_index()
                .index
            )
            missing = list(set(orig_vals) - set(getattr(df_datahist, a)))
            imissing = []
            for v in missing:
                imissing.append(np.where(orig_vals == v)[0][0])

            vals = [
                "{1:.{0}f}_{2:.{0}f}".format(
                    ndecimals, binning_dict[c][i][0], binning_dict[c][i][1]
                )
                for i in range(len(binning_dict[c]))
                if not i in imissing
            ]
            vals += list(getattr(df_datahist, a)[len(vals) :])  # to deal with UF and OF
            setattr(df_datahist, a, vals)

    pvalues, zvalues = outlier_significance_matrix_from_hist2d(
        df_datahist.values, CI_method=CI_method
    )
    outlier_overview = pd.DataFrame(
        zvalues, index=df_datahist.index, columns=df_datahist.columns
    )

    return outlier_overview


def outlier_significance_matrix(
    df: pd.DataFrame,
    interval_cols: Optional[list] = None,
    CI_method: str = "poisson",
    ndecimals: int = 1,
    bins=10,
    quantile: bool = False,
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
    retbins: bool = False,
    verbose: bool = True,
):
    """
    Calculate the significance matrix of excesses or deficits

    :param df: input data. DataFrame must contain exactly two columns
    :param interval_cols: columns with interval variables which need to be binned
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error.\
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param ndecimals: number of decimals to use in labels of binned interval variables to specify bin edges (default=1)
    :param bool quantile: when the number of bins is specified, use uniform binning (False) or quantile binning (True)
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :param bool retbins: if true, function also returns dict with bin_edges of rebinned variables.
    :param bool verbose: if False, do not print all interval columns that are guessed
    :return: outlier significance matrix (pd.DataFrame)
    """

    if len(df.columns) != 2:
        raise ValueError("df should contain only two columns")

    if interval_cols is None:
        interval_cols = guess_interval_cols(df, verbose)

    df_clean, interval_cols_clean = dq_check_nunique_values(
        df, interval_cols, dropna=dropna
    )

    data_binned, binning_dict = bin_data(
        df_clean, interval_cols_clean, retbins=True, bins=bins, quantile=quantile
    )

    os_matrix = outlier_significance_matrix_from_rebinned_df(
        data_binned,
        binning_dict,
        CI_method=CI_method,
        ndecimals=ndecimals,
        dropna=dropna,
        drop_underflow=drop_underflow,
        drop_overflow=drop_overflow,
    )

    if retbins:
        return os_matrix, binning_dict
    return os_matrix


def outlier_significance_matrices_from_rebinned_df(
    data_binned: pd.DataFrame,
    binning_dict=None,
    CI_method="poisson",
    ndecimals=1,
    combinations: Union[list, tuple] = (),
    dropna=True,
    drop_underflow=True,
    drop_overflow=True,
):
    """
    Calculate the significance matrix of excesses or deficits for all possible combinations of variables, or for
    those combinations specified using combinations. This functions could also be used instead of
    outlier_significance_matrices in case all variables are either categorical or ordinal, so no binning is required.

    :param data_binned: input data. Interval variables need to be binned. DataFrame must contain exactly two columns
    :param dict binning_dict: dictionary with bin edges for each binned interval variable. When no bin_edges are\
    provided values are used as bin label. Otherwise, bin labels are constructed based on provided bin edge information.
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error.\
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param ndecimals: number of decimals to use in labels of binned interval variables to specify bin edges (default=1)
    :param combinations: in case you do not want to calculate an outlier significance matrix for all permutations of\
    the available variables, you can specify a list of the required permutations here, in the format\
    [(var1, var2), (var2, var4), etc]
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :return: dictionary with outlier significance matrices (pd.DataFrame)
    """
    if binning_dict is None:
        binning_dict = {}

    # create a list of all possible combinations of variables, in case no selection of combinations is specified
    if not combinations:
        combinations = itertools.combinations(data_binned.columns, 2)

    outliers_overview = []
    for i, (c0, c1) in enumerate(combinations):
        zvalues_overview = outlier_significance_matrix_from_rebinned_df(
            data_binned[[c0, c1]].copy(),
            binning_dict,
            CI_method=CI_method,
            ndecimals=ndecimals,
            dropna=dropna,
            drop_underflow=drop_underflow,
            drop_overflow=drop_overflow,
        )
        outliers_overview.append((c0, c1, zvalues_overview))

    return outliers_overview


def outlier_significance_matrices(
    df: pd.DataFrame,
    interval_cols: Optional[list] = None,
    CI_method: str = "poisson",
    ndecimals: int = 1,
    bins=10,
    quantile: bool = False,
    combinations: Union[list, tuple] = (),
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
    retbins: bool = False,
    verbose: bool = True,
):
    """
    Calculate the significance matrix of excesses or deficits for all possible combinations of variables, or for
    those combinations specified using combinations

    :param df: input data
    :param interval_cols: columns with interval variables which need to be binned
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error. \
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param ndecimals: number of decimals to use in labels of binned interval variables to specify bin edges (default=1)
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param bool quantile: when the number of bins is specified, use uniform binning (False) or quantile binning (True)
    :param combinations: in case you do not want to calculate an outlier significance matrix for all permutations of\
    the available variables, you can specify a list of the required permutations here, in the format\
    [(var1, var2), (var2, var4), etc]
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :param bool retbins: if true, function also returns dict with bin_edges of rebinned variables.
    :param bool verbose: if False, do not print all interval columns that are guessed
    :return: dictionary with outlier significance matrices (pd.DataFrame)
    """

    if interval_cols is None:
        interval_cols = guess_interval_cols(df, verbose)

    df_clean, interval_cols_clean = dq_check_nunique_values(
        df, interval_cols, dropna=dropna
    )

    data_binned, binning_dict = bin_data(
        df_clean, interval_cols_clean, retbins=True, bins=bins, quantile=quantile
    )

    os_matrices = outlier_significance_matrices_from_rebinned_df(
        data_binned,
        binning_dict,
        CI_method,
        ndecimals,
        combinations=combinations,
        dropna=dropna,
        drop_underflow=drop_underflow,
        drop_overflow=drop_overflow,
    )

    # Convert to dict
    os_matrices = {":".join([c0, c1]): v for c0, c1, v in os_matrices}

    if retbins:
        return os_matrices, binning_dict
    return os_matrices


def outlier_significance_from_array(
    x: Union[np.ndarray, list, pd.Series],
    y: Union[np.ndarray, list, pd.Series],
    num_vars: list = None,
    bins: Union[int, list, np.ndarray, dict] = 10,
    quantile: bool = False,
    ndecimals: int = 1,
    CI_method: str = "poisson",
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculate the significance matrix of excesses or deficits of input x and input y. x and y can contain interval, \
    ordinal or categorical data. Use the num_vars variable to indicate whether x and/or y contain interval data.

    :param list x: array-like input
    :param list y: array-like input
    :param list num_vars: list of variables which are numeric and need to be binned, either ['x'],['y'],or['x','y']
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param bool quantile: when the number of bins is specified, use uniform binning (False) or quantile binning (True)
    :param ndecimals: number of decimals to use in labels of binned interval variables to specify bin edges (default=1)
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error. \
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning a \
    numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True \
    (relevant when binning a numeric variable)
    :param bool verbose: if False, do not print all interval columns that are guessed
    :return: outlier significance matrix (pd.DataFrame)
    """

    df = array_like_to_dataframe(x, y)

    if num_vars is None:
        num_vars = guess_interval_cols(df, verbose)

    return outlier_significance_matrix(
        df,
        interval_cols=num_vars,
        bins=bins,
        quantile=quantile,
        ndecimals=ndecimals,
        CI_method=CI_method,
        dropna=dropna,
        drop_underflow=drop_underflow,
        drop_overflow=drop_overflow,
        verbose=verbose,
    )


def outlier_significance_from_binned_array(
    x: Union[np.ndarray, list, pd.Series],
    y: Union[np.ndarray, list, pd.Series],
    CI_method: str = "poisson",
    dropna: bool = True,
    drop_underflow: bool = True,
    drop_overflow: bool = True,
) -> pd.DataFrame:

    """
    Calculate the significance matrix of excesses or deficits of input x and input y. x and y can contain binned
    interval, ordinal or categorical data.

    :param list x: array-like input
    :param list y: array-like input
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error. \
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning \
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning \
    a numeric variable)
    :return: outlier significance matrix (pd.DataFrame)
    """

    df = array_like_to_dataframe(x, y)

    return outlier_significance_matrix(
        df,
        CI_method=CI_method,
        dropna=dropna,
        drop_underflow=drop_underflow,
        drop_overflow=drop_overflow,
    )
