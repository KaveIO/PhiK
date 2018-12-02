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

from typing import Tuple, Union
import itertools
import numpy as np
import pandas as pd

from scipy import stats
from scipy.special import betainc

from phik import definitions as defs
from .binning import bin_data
from .betainc import log_incompbeta
from .statistics import z_from_logp


def poisson_obs_p(nobs:int, nexp:float, nexperr:float) -> float:
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
        nexpalt = nexp if nexp>0 else nexperr
        tau = nexpalt/(nexperr*nexperr)
        b = nexpalt*tau+1
        x = 1/(1+tau)
        p = betainc(nobs, b, x)
    else: # assume error==0
        p = stats.poisson.sf(nobs-1, nexp)

    return p


def log_poisson_obs_p(nobs:int, nexp:float, nexperr:float) -> Tuple[float,float]:
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
        return (0, -np.inf)

    if nexperr > 0:
        nexpalt = nexp if nexp>0 else nexperr
        tau = nexpalt/(nexperr*nexperr)
        b = nexpalt*tau+1
        x = 1/(1+tau)
        tlogp = log_incompbeta(nobs, b, x)
    else: # assume error==0. nobs>0 at this stage
        logp = stats.poisson.logsf(nobs-1, nexp)
        p = stats.poisson.sf(nobs-1, nexp)
        tlogp = (logp, np.log(1-p))

    return tlogp


def poisson_obs_z(nobs:int, nexp:float, nexperr:float) -> float:
    """
    Calculate the Z-value for measuring nobs observations given the expected value.

    The Z-value express the number
    of sigmas the observed value diviates from the expected value, and is based on the p-value calculation. If the uncertainty
    on the expected value is known the Linnemann method is used. Otherwise the Poisson distribution is used to estimate the p-value.

    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: Z-value
    :rtype: float
    """
    pvalue = poisson_obs_p(nobs, nexp, nexperr)

    # special cases: numerically too close to zero or one.
    # try to evaluate log(p) or log(1-p)
    if pvalue==0 or pvalue==1:
        tlogp = log_poisson_obs_p(nobs, nexp, nexperr)
        if pvalue==0:
            logp = tlogp[0]
            Z = z_from_logp(logp)
        if pvalue==1:
            log1mp = tlogp[1]
            Z = z_from_logp(logp, flip_sign = True)
    # default:
    else:
        Z = -stats.norm.ppf(pvalue)

    return Z


def poisson_obs_mid_p(nobs:int, nexp:float, nexperr:float) -> float:
    """
    Calculate the p-value for measuring nobs observations given the expected value.

    The Lancaster mid-P correction is
    applied to take into account the effects of discrete statistics. If the uncertainty on the expected value is known the
    Linnemann method is used for the p-value calcuation. Otherwise the Poisson distribution is used to estimate the p-value.
    
    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: mid p-value
    :rtype: float
    """
    p = poisson_obs_p(nobs, nexp, nexperr)
    pplus1 = poisson_obs_p(nobs+1, nexp, nexperr)
    mid_p = 0.5 * (p - pplus1)
    p -= mid_p

    return p


def log_poisson_obs_mid_p(nobs:int, nexp:float, nexperr:float) -> Tuple[float,float]:
    """
    Calculate the logarithm of the p-value for measuring nobs observations given the expected value.

    The Lancaster mid-P correction is
    applied to take into account the effects of discrete statistics. If the uncertainty on the expected value is known the
    Linnemann method is used for the p-value calcuation. Otherwise the Poisson distribution is used to estimate the p-value.

    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: tuple of log(p) and log(1-p)
    :rtype: tuple
    """
    tlogp = log_poisson_obs_p(nobs, nexp, nexperr)
    tlogpp1 = log_poisson_obs_p(nobs+1, nexp, nexperr)

    # 1. evaluate log([p+pp1]/2) ; note that p > pp1
    #    = log(0.5) + log(p) + log(1 + exp[log(pp1)-log(p)])
    lp = tlogp[0]
    lp1 = tlogpp1[0]
    logmidp = np.log(0.5) + lp + np.log( 1 + np.exp(lp1-lp) )

    # 2. let q = 1 - p; note that qp1 > q
    #    evaluate log(1-[p+pp1]/2) = log ([q+qp1]/2)
    #    = log(0.5) + log(qp1) + log(1 + exp[log(q)-log(qp1)])
    lq = tlogp[1]
    lq1 = tlogpp1[1]
    logmidq = np.log(0.5) + lq1 + np.log( 1 + np.exp(lq-lq1) )

    return (logmidp, logmidq)


def poisson_obs_mid_z(nobs:int, nexp:float, nexperr:float) -> float:
    """Calculate the Z-value for measuring nobs observations given the expected value.

    The Z-value express the number
    of sigmas the observed value diviates from the expected value, and is based on the p-value calculation.
    The Lancaster midP correction is applied to take into account the effects of low statistics. If the uncertainty on the 
    expected value is known the Linnemann method is used for the p-value calcuation. Otherwise the Poisson distribution is 
    used to estimate the p-value.

    :param int nobs: observed count
    :param float nexp: expected number
    :param float nexperr: uncertainty on the expected number
    :returns: Z-value
    :rtype: tuple
    """
    pvalue = poisson_obs_mid_p(nobs, nexp, nexperr)

    # special cases: numerically too close to zero or one.
    # try to evaluate log(p) or log(1-p)
    if pvalue==0 or pvalue==1:
        tlogp = log_poisson_obs_mid_p(nobs, nexp, nexperr)
        if pvalue==0:
            logp = tlogp[0]
            Z = z_from_logp(logp)
        if pvalue==1:
            log1mp = tlogp[1]
            Z = z_from_logp(log1mp, flip_sign = True)
    # default:
    else:
        Z = -stats.norm.ppf(pvalue)

    return Z


def get_independent_frequency_estimates(values:np.ndarray, CI_method:str='poisson') -> Union[np.ndarray, np.ndarray]:
    """
    Calculation of expected frequencies, based on the ABCD-method, i.e. independent frequency estimates.

    :param values: The contingency table. The table contains the observed number of occurrences in each category.
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error.\
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :returns exp, experr: expected frequencies, error on the expected frequencies
    """
    if not isinstance(values, np.ndarray):
        raise TypeError('values is not a numpy array.')

    # Initialize
    exp = np.zeros(values.shape)
    experr = np.zeros(values.shape)

    # Calculate dependent expected value using ABCD method
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            Aobs = values[i][j]
            B = values[i].sum() - Aobs
            C = values[:,j].sum() - Aobs
            D = values.sum() - B - C - Aobs
            # prediction for A can only be calculated if D is non-zero
            if D > 0:
                exp[i][j] = B*C/D
                sigmaB = get_uncertainty(B, CI_method=CI_method)
                sigmaC = get_uncertainty(C, CI_method=CI_method)
                sigmaD = get_uncertainty(D, CI_method=CI_method)
                experr[i][j] = np.sqrt(pow(sigmaB*C/D,2) + pow(sigmaC*B/D,2) + pow(sigmaD*exp[i][j]/D,2))
            # in case of zero D, A is infinity. Set prediction to NaN.
            else:
                exp[i][j] = np.nan
                experr[i][j] = np.nan

    return exp, experr


def get_uncertainty(x:float, CI_method:str='poisson') -> float:
    """
    Calculate the uncertainty on a random number x taken from the poisson distribution

    The uncertainty on the x is calculated using either the standard poisson error (poisson) or using the asymmetric
    exact poisson interval (exact_poisson).
    https://www.ncbi.nlm.nih.gov/pubmed/2296988 #FIXME: check ref

    :param float x: value
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error.\
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :return xerr: the uncertainty on x (1 sigma)
    """    
    assert CI_method in ['exact_poisson', 'poisson'], 'CI method %s not valid' % CI_method
    assert x>=0, 'x must be equal or greater than zero'

    if CI_method=='exact_poisson':
        xerr = get_exact_poisson_uncertainty(x)
    if CI_method=='poisson':
        xerr = get_poisson_uncertainty(x)

    return xerr


def get_poisson_uncertainty(x:float) -> float:
    """
    Calculate the uncerainty on x using standard poisson error. In case x=0 the error=1 is assigned.

    :param float x: value
    :return xerr: the uncertainty on x (1 sigma)
    :rtype: float
    """
    err = np.sqrt(x) if x>=1 else 1.0
    return err


def get_exact_poisson_uncertainty(x:float, nsigmas:float=1) -> float:
    """
    Calculate the uncerainty on x using an exact poisson confidence interval.

    Calculate the uncerainty on x using an exact poisson confidence interval. The width of the confidence interval can 
    be specified using the number of sigmas. The default number of sigmas is set to 1, resulting in an error that is 
    approximated by the standard poisson error sqrt(x).

    Exact poisson uncertainty is described here:
    https://ms.mcmaster.ca/peter/s743/poissonalpha.html
    https://www.statsdirect.com/help/rates/poisson_rate_ci.htm
    https://www.ncbi.nlm.nih.gov/pubmed/2296988

    :param float x: value
    :return xerr: the uncertainty on x (1 sigma)
    :rtype: float
    """
    # see formula at:
    # https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
    pl = stats.norm.cdf(-1*nsigmas, loc=0, scale=1)
    pu = stats.norm.cdf(1*nsigmas, loc=0, scale=1)

    lb = stats.chi2.ppf(pl, 2*x)/2 if x!= 0 else 0
    ub = stats.chi2.ppf(pu, 2*(x+1))/2 

    # average err is almost equal to sqrt(x)+0.5
    err = (ub-lb)/2

    return err


def get_outlier_significances(obs:np.ndarray, exp:np.ndarray, experr:np.ndarray) -> Union[np.ndarray, np.ndarray]:
    """
    Evaluation of significance of observation

    Evaluation of the significance of the difference between the observed number of occurences and the expected number of 
    occurences, taking into account the uncertainty on the expectednumber of occurences. When the uncertainty is 
    not zero, the Linnemann method is used to calculate the pvalues.

    :param obs: observed numbers
    :param exp: expected numbers
    :param experr: uncertainty on the expected numbers
    :returns: pvalues, zvalues
    """
    if not isinstance(obs, np.ndarray):
        raise TypeError('obs is not a numpy array.')
    if not isinstance(exp, np.ndarray):
        raise TypeError('exp is not a numpy array.')
    if not isinstance(experr, np.ndarray):
        raise TypeError('experr is not a numpy array.')

    pvalues = np.zeros(obs.shape)
    zvalues = np.zeros(obs.shape)
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            pvalues[i][j] = poisson_obs_mid_p(obs[i][j], exp[i][j], experr[i][j])
            zvalues[i][j] = poisson_obs_mid_z(obs[i][j], exp[i][j], experr[i][j])

    return pvalues, zvalues


def outlier_significance_matrix_from_hist2d(data:np.ndarray, CI_method:str='poisson') -> Union[np.ndarray, np.ndarray]:
    """
    Calculate the significance matrix of excesses or deficits in a contingency table

    :param data: numpy array contingency table
    :param string CI_method: method to be used for undertainty calculation. poisson: normal poisson error.\
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :return: pvalue matrix, outlier significance matrix
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data is not a numpy array.')

    # get expected values
    exp, experr = get_independent_frequency_estimates(data, CI_method=CI_method)
    pvalues, zvalues = get_outlier_significances(data, exp, experr)

    return pvalues, zvalues


def outlier_significance_matrix_from_rebinned_df(data_binned:pd.DataFrame, binning_dict:dict, CI_method:str='poisson', ndecimals:int=1,
                                                 dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True) -> pd.DataFrame:
    """
    Calculate the significance matrix of excesses or deficits

    :param data_binned: input data. Dataframe must contain exactly two columns
    :param dict binning_dict: dictionary with bin edges for each binned interval variable. When no bin_edges are\
    provided values are used as bin label. Otherwise, bin labels are constructed based on provided bin edge information.
    :param string CI_method: method to be used for undertainty calculation. poisson: normal poisson error. \
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param ndecimals: number of decimals to use in labels of binned interval variables to specify bin edges (default=1)
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning\
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning\
    a numeric variable)
    :return: outlier significance matrix (pd.DataFrame)
    """
    if not isinstance(data_binned, pd.DataFrame):
        raise TypeError('data_binned is not a pandas DataFrame.')
    assert len(data_binned.columns) == 2, 'data DataFrame should contain only two columns'

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

    if c0 in binning_dict.keys():
        #index_vals = ['{1:.{0}f}_{2:.{0}f}'.format(ndecimals, binning_dict[c0][i], binning_dict[c0][i+1])
        #              for i in range(len(binning_dict[c0])-1)]
        index_vals = ['{1:.{0}f}_{2:.{0}f}'.format(ndecimals, binning_dict[c0][i][0], binning_dict[c0][i][1])
                      for i in range(len(binning_dict[c0]))]
        index_vals = index_vals + list(df_datahist.index[len(index_vals):])  # to deal with UF and OF
        df_datahist.index = index_vals

    if c1 in binning_dict.keys():
        #col_vals = ['{1:.{0}f}_{2:.{0}f}'.format(ndecimals, binning_dict[c1][i], binning_dict[c1][i+1])
        #            for i in range(len(binning_dict[c1])-1)]
        col_vals = ['{1:.{0}f}_{2:.{0}f}'.format(ndecimals, binning_dict[c1][i][0], binning_dict[c1][i][1])
                    for i in range(len(binning_dict[c1]))]
        col_vals = col_vals + list(df_datahist.columns[len(col_vals):])  # to deal with UF and OF
        df_datahist.columns = col_vals

    pvalues, zvalues = outlier_significance_matrix_from_hist2d(df_datahist.values, CI_method=CI_method)
    outlier_overview = pd.DataFrame(zvalues, index=df_datahist.index, columns=df_datahist.columns)

    return outlier_overview


def outlier_significance_matrix(df:pd.DataFrame, interval_cols:list=None, CI_method:str='poisson', ndecimals:int=1,
                                bins=10, quantile:bool=False,
                                dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True, retbins:bool=False):
    """
    Calculate the significance matrix of excesses or deficits

    :param df: input data. Dataframe must contain exactly two columns
    :param interval_cols: columns with interval variables which need to be binned
    :param string CI_method: method to be used for undertainty calculation. poisson: normal poisson error.\
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
    :return: outlier significance matrix (pd.DataFrame)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df is not a pandas DataFrame.')
    assert len(df.columns) == 2, 'df should contain only two columns'

    if isinstance( interval_cols, type(None) ):
        interval_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if interval_cols:
            print('interval_cols not set, guessing: {0:s}'.format(str(interval_cols)))
    assert isinstance( interval_cols, list ), 'interval_cols is not a list.'

    data_binned, binning_dict = bin_data(df, interval_cols, retbins=True, bins=bins, quantile=quantile)

    os_matrix = outlier_significance_matrix_from_rebinned_df(data_binned, binning_dict, CI_method=CI_method,
                                                             ndecimals=ndecimals, dropna=dropna,
                                                             drop_underflow=drop_underflow, drop_overflow=drop_overflow)
    if retbins:
        return os_matrix, binning_dict
    return os_matrix


def outlier_significance_matrices_from_rebinned_df(data_binned, binning_dict={}, CI_method='poisson', ndecimals=1,
                                                   combinations=[], dropna=True, drop_underflow=True,
                                                   drop_overflow=True):
    """
    Calculate the significance matrix of excesses or deficits for all possible combinations of variables, or for
    those combinations specified using combinations. This functions could also be used instead of
    outlier_significance_matrices in case all variables are either categorical or ordinal, so no binning is required.

    :param data_binned: input data. Interval variables need to be binned. Dataframe must contain exactly two columns
    :param dict binning_dict: dictionary with bin edges for each binned interval variable. When no bin_edges are\
    provided values are used as bin label. Otherwise, bin labels are constructed based on provided bin edge information.
    :param string CI_method: method to be used for uncertainty calculation. poisson: normal poisson error.\
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param bins: specify the binning, either by proving the number of bins, a list of bin edges, or a dictionary with\
    bin specifications per variable. (default=10)
    :param ndecimals: number of decimals to use in labels of binned interval variables to specify bin edges (default=1)
    :param bool quantile: when the number of bins is specified, use uniform binning (False) or quantile binning (True)
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
    if not isinstance(data_binned, pd.DataFrame):
        raise TypeError('data_binned is not a pandas DataFrame.')

    # create a list of all possible combinations of variables, in case no selection of combinations is specified
    if not combinations:
        combinations = itertools.combinations(data_binned.columns, 2)

    outliers_overview = {}
    for i, comb in enumerate(combinations):
        c0, c1 = comb
        zvalues_overview = outlier_significance_matrix_from_rebinned_df(data_binned[[c0, c1]], binning_dict,
                                                                        CI_method=CI_method, ndecimals=ndecimals,
                                                                        dropna=dropna, drop_underflow=drop_underflow,
                                                                        drop_overflow=drop_overflow)
        outliers_overview[':'.join(comb)] = zvalues_overview

    return outliers_overview


def outlier_significance_matrices(df:pd.DataFrame, interval_cols:list=None, CI_method:str='poisson', ndecimals:int=1, bins=10,
                                  quantile:bool=False, combinations:list=[],
                                  dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True, retbins:bool=False):
    """
    Calculate the significance matrix of excesses or deficits for all possible combinations of variables, or for
    those combinations specified using combinations

    :param df: input data
    :param interval_cols: columns with interval variables which need to be binned
    :param string CI_method: method to be used for undertainty calculation. poisson: normal poisson error. \
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
    :return: dictionary with outlier significance matrices (pd.DataFrame)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df is not a pandas DataFrame.')

    if isinstance(interval_cols, type(None)):
        interval_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if interval_cols:
            print('interval_cols not set, guessing: {0:s}'.format(str(interval_cols)))
    assert isinstance(interval_cols, list), 'interval_cols is not a list.'

    data_binned, binning_dict = bin_data(df, interval_cols, retbins=True, bins=bins, quantile=quantile)

    os_matrices = outlier_significance_matrices_from_rebinned_df(data_binned, binning_dict, CI_method, ndecimals,
                                                                 combinations=combinations, dropna=dropna,
                                                                 drop_underflow=drop_underflow,
                                                                 drop_overflow=drop_overflow)
    if retbins:
        return os_matrices, binning_dict
    return os_matrices


def outlier_significance_from_array(x, y, num_vars:list=None, bins=10, quantile:bool=False, ndecimals:int=1, CI_method:str='poisson',
                                    dropna:bool=True, drop_underflow:bool=True, drop_overflow:bool=True) -> pd.DataFrame:
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
    :param string CI_method: method to be used for undertainty calculation. poisson: normal poisson error. \
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning a \
    numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True \
    (relevant when binning a numeric variable)
    :return: outlier significance matrix (pd.DataFrame)
    """
    if not isinstance(x, (np.ndarray, list, pd.Series)):
        raise TypeError('x is not array like.')
    if not isinstance(y, (np.ndarray, list, pd.Series)):
        raise TypeError('y is not array like.')
    if not isinstance(bins, (int,list,np.ndarray,dict)):
        raise TypeError('bins is of incorrect type.')    

    df = pd.DataFrame(np.array([x, y]).T, columns=['x', 'y'])

    if isinstance( num_vars, type(None) ):
        num_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_vars:
            print('num_vars not set, guessing: {0:s}'.format(str(num_vars)))
    assert isinstance( num_vars, list ), 'num_vars is not a list.'

    return outlier_significance_matrix(df, interval_cols=num_vars, bins=bins, quantile=quantile, ndecimals=ndecimals,
                                       CI_method=CI_method,
                                       dropna=dropna, drop_underflow=drop_underflow, drop_overflow=drop_overflow)


def outlier_significance_from_binned_array(x, y, CI_method:str='poisson', dropna:bool=True, drop_underflow:bool=True,
                                           drop_overflow:bool=True) -> pd.DataFrame:

    """
    Calculate the significance matrix of excesses or deficits of input x and input y. x and y can contain binned
    interval, ordinal or categorical data.

    :param list x: array-like input
    :param list y: array-like input
    :param string CI_method: method to be used for undertainty calculation. poisson: normal poisson error. \
    exact_poisson: error calculated from the asymmetric exact poisson interval
    :param bool dropna: remove NaN values with True
    :param bool drop_underflow: do not take into account records in underflow bin when True (relevant when binning \
    a numeric variable)
    :param bool drop_overflow: do not take into account records in overflow bin when True (relevant when binning \
    a numeric variable)
    :return: outlier significance matrix (pd.DataFrame)
    """
    if not isinstance(x, (np.ndarray, list, pd.Series)):
        raise TypeError('x is not array like.')
    if not isinstance(y, (np.ndarray, list, pd.Series)):
        raise TypeError('y is not array like.')

    df = pd.DataFrame(np.array([x, y]).T, columns=['x', 'y'])
    return outlier_significance_matrix(df, CI_method=CI_method, dropna=dropna, drop_underflow=drop_underflow,
                                       drop_overflow=drop_overflow)
