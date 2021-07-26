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
from typing import Union

import numpy as np
from scipy import stats


def get_dependent_frequency_estimates(vals: np.ndarray) -> np.ndarray:
    """
    Calculation of dependent expected frequencies.

    Calculation is based on the marginal sums of the table, i.e. dependent frequency estimates.
    :param vals: The contingency table. The table contains the observed number of occurrences in each category

    :returns exp: expected frequencies
    """

    # use existing scipy functionality
    return stats.contingency.expected_freq(vals)


def get_chi2_using_dependent_frequency_estimates(
    vals: np.ndarray, lambda_: str = "log-likelihood"
) -> float:
    """
    Chi-square test of independence of variables in a contingency table.

    The expected frequencies are based on the
    marginal sums of the table, i.e. dependent frequency estimates.

    :param vals: The contingency table. The table contains the observed number of occurrences in each category
    :returns test_statistic: the test statistic value
    """

    values = vals[:]

    # remove rows with only zeros, scipy doesn't like them.
    values = values[~np.all(values == 0, axis=1)]
    # remove columns with only zeros, scipy doesn't like them.
    values = values.T[~np.all(values.T == 0, axis=1)].T

    # use existing scipy functionality
    test_statistic, _, _, _ = stats.chi2_contingency(values, lambda_=lambda_)

    return test_statistic


def get_pearson_chi_square(
    observed: np.ndarray, expected: np.ndarray = None, normalize: bool = True
) -> float:
    """Calculate pearson chi square between observed and expected 2d contingency matrix

    :param observed: The observed contingency table. The table contains the observed number of occurrences in each cell.
    :param expected: The expected contingency table. The table contains the expected number of occurrences in each cell.
    :param bool normalize: normalize expected frequencies, default is True.
    :return: the pearson chi2 value
    """
    observed = np.asarray(observed)
    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be non-negative.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")

    if expected is None:
        expected = get_dependent_frequency_estimates(observed)
    expected = np.asarray(expected)

    # important to ensure that observed and expected have same normalization
    if normalize:
        expected = expected * (np.sum(observed) / np.sum(expected))

    terms = np.divide(
        (observed.astype(np.float64) - expected) ** 2,
        expected,
        out=np.zeros_like(expected),
        where=expected != 0,
    )
    return np.sum(terms)


def estimate_ndof(chi2values: Union[list, np.ndarray]) -> float:
    """
    Estimation of the effective number of degrees of freedom.

    A good approximation of endof is the average value. Alternatively
    a fit to the chi2 distribution can be make. Both values are returned.

    :param list chi2values: list of chi2 values
    :returns: endof0, endof
    """

    return np.mean(chi2values)


def estimate_simple_ndof(observed: np.ndarray) -> int:
    """
    Simple estimation of the effective number of degrees of freedom.

    This equals the nominal calculation for ndof minus the number of empty bins in the
    expected contingency table.

    :param observed: numpy array of observed cell counts
    :returns: endof
    """

    # use existing scipy functionality
    expected = stats.contingency.expected_freq(observed)
    endof = (
        expected.size
        - np.sum(expected.shape)
        + expected.ndim
        - 1
        - (expected == 0).sum()
    )
    # require minimum number of degrees of freedom
    if endof < 0:
        endof = 0
    return endof


def theoretical_ndof(observed: np.ndarray) -> int:
    """
    Simple estimation of the effective number of degrees of freedom.

    This equals the nominal calculation for ndof minus the number of empty bins in the
    expected contingency table.

    :param observed: numpy array of observed cell counts
    :returns: theoretical ndof
    """

    return observed.size - np.sum(observed.shape) + observed.ndim - 1


def z_from_logp(logp: float, flip_sign: bool = False) -> float:
    """
    Convert logarithm of p-value into one-sided Z-value

    :param float logp: logarithm of p-value, should not be greater than 0
    :param bool flip_sign: flip sign of Z-value, e.g. use for input log(1-p). Default is false.
    :returns: statistical significance Z-value
    :rtype: float
    """

    # pvalue == 0, Z = infinity
    if logp == -np.inf:
        return np.inf if not flip_sign else -np.inf

    p_value = np.exp(logp)

    # scenario where p-value is numerically too small to evaluate Z
    if p_value == 0:
        # kicks in here when Z > 37
        # approach valid when ~ Z > 1.5.
        u = -2.0 * np.log(2 * np.pi) - 2.0 * logp
        z_value = np.sqrt(u - np.log(u))
    else:
        z_value = -stats.norm.ppf(p_value)

    if flip_sign:
        z_value *= -1.0

    return z_value
