"""Project: PhiK - correlation coefficient package

Created: 2018/09/05

Description:
    Statistics helper functions, for the calculation of phik and significance
    of a contingency table.

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import copy
import numpy as np
from scipy import stats


def get_dependent_frequency_estimates(vals:np.ndarray) -> np.ndarray:
    """
    Calculation of dependent expected frequencies.

    Calculation is based on the marginal sums of the table, i.e. dependent frequency estimates.
    :param values: The contingency table. The table contains the observed number of occurrences in each category

    :returns exp: expected frequencies 
    """
    if not isinstance(vals, np.ndarray):
        raise TypeError('vals is not a numpy array.')

    # use existing scipy functionality
    return stats.contingency.expected_freq(vals)


def get_chi2_using_dependent_frequency_estimates(vals:np.ndarray, lambda_:str = 'log-likelihood') -> float:
    """
    Chi-square test of independence of variables in a contingency table.

    The expected frequencies are based on the
    marginal sums of the table, i.e. dependent frequency estimates.

    :param values: The contingency table. The table contains the observed number of occurrences in each category
    :returns chi2: 
    """
    if not isinstance(vals, np.ndarray):
        raise TypeError('vals is not a numpy array.')

    values = copy.copy(vals)

    # create np array
    if type(values) == list:
        values=np.array(values)

    # remove rows with only zeros, scipy doesn't like them.
    values = values[~np.all(values == 0, axis=1)]
    # remove columns with only zeros, scipy doesn't like them.
    values = values.T[~np.all(values.T == 0, axis=1)].T

    # use existing scipy functionality
    exp = stats.chi2_contingency(values, lambda_=lambda_)

    return exp[0]


def estimate_ndof(chi2values:list) -> float:
    """
    Estimation of the effective number of degrees of freedom.

    A good approximation of endof is the average value. Alternatively
    a fit to the chi2 distribution can be make. Both values are returned.

    :param list chi2values: list of chi2 values
    :returns: endof0, endof
    """
    if not isinstance(chi2values, (np.ndarray, list)):
        raise TypeError('chi2values is not array like.')

    endof0 = np.mean(chi2values)
    return endof0


def estimate_simple_ndof(observed:np.ndarray) -> int:
    """
    Simple estimation of the effective number of degrees of freedom.

    This equals the nominal calculation for ndof minus the number of empty bins in the 
    expected contingency table.

    :param observed: numpy array of observed cell counts
    :returns: endof
    """
    if not isinstance(observed, np.ndarray):
        raise TypeError('observed is not a numpy array.')

    # use existing scipy functionality
    expected = stats.contingency.expected_freq(observed)
    endof = expected.size - np.sum(expected.shape) + expected.ndim - 1 - (expected==0).sum()
    # require minimum number of degrees of freedom
    if endof < 0:
        endof = 0
    return endof


def theoretical_ndof(observed:np.ndarray) -> int:
    """
    Simple estimation of the effective number of degrees of freedom.

    This equals the nominal calculation for ndof minus the number of empty bins in the 
    expected contingency table.

    :param observed: numpy array of observed cell counts
    :returns: theoretical ndof
    """    
    if not isinstance(observed, np.ndarray):
        raise TypeError('observed is not a numpy array.')

    ndof = observed.size - np.sum(observed.shape) + observed.ndim - 1
    return ndof


def z_from_logp(logp:float, flip_sign:bool = False) -> float:
    """
    Convert logarithm of p-value into one-sided Z-value

    :param float logp: logarithm of p-value
    :param bool flip_sign: flip sign of Z-value, e.g. use for input log(1-p). Default is false.
    :returns: statistical significance Z-value
    :rtype: float
    """
    if logp > 0:
        raise ValueError('logp={:f} cannot be greater than zero'.format(logp))

    # pvalue == 0, Z = infinity
    if logp == -np.inf:
        return np.inf if not flip_sign else -np.inf

    pvalue = np.exp(logp)

    # scenario where pvalue is numerically too small to evaluate Z
    if pvalue == 0:
        # kicks in here when Z > 37
        # approach valid when ~ Z > 1.5.
        u = -2.*np.log(2 * np.pi) - 2.*logp
        Zvalue = np.sqrt(u - np.log(u))
    else:
        Zvalue = -stats.norm.ppf(pvalue)

    if flip_sign:
        Zvalue *= -1.

    return Zvalue
