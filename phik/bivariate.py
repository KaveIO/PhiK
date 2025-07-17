"""Project: PhiK - correlation analyzer library

Created: 2019/11/23

Description:
    Convert Pearson correlation value into a chi2 value of a contingency test 
    matrix of a bivariate gaussian, and vice-versa.
    Calculation uses scipy's mvn library.

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""
import warnings

import numpy as np
import scipy
from scipy import optimize

_scipy_version = [int(v) for v in scipy.__version__.split('.')]
USE_QMVN = True if _scipy_version[0] >= 1 and _scipy_version[1] >= 16 else False
if USE_QMVN:
    from scipy.stats._qmvnt import _qauto, _qmvn
else:
    from scipy.stats._mvn import mvnun




def _mvn_un(rho: float, lower: tuple, upper: tuple,
            rng: np.random.Generator = np.random.default_rng(42)) -> float:
    """Perform integral of bivariate normal gauss with correlation

    Integral is performed using scipy's mvn library.

    :param float rho: tilt parameter
    :param tuple lower: tuple of lower corner of integral area
    :param tuple upper: tuple of upper corner of integral area
    :param np.random.Generator rng: default_rng(42), optional
    :returns float: integral value
    """
    mu = np.array([0.0, 0.0])
    S = np.array([[1.0, rho], [rho, 1.0]])
    return _calc_mvnun(lower=lower, upper=upper, mu=mu, S=S, rng=rng)


def _calc_mvnun(lower, upper, mu, S, rng = np.random.default_rng(42)):
    if USE_QMVN:
        res = _qauto(_qmvn, S, lower, upper, rng)[0]
    else:
        res = mvnun(lower, upper, mu, S)[0]
    return res


def _mvn_array(rho: float, sx: np.ndarray, sy: np.ndarray) -> list:
    """Array of integrals over bivariate normal gauss with correlation

    Integrals are performed using scipy's mvn library.

    :param float rho: tilt parameter
    :param np.ndarray sx: bin edges array of x-axis
    :param np.ndarray sy: bin edges array of y-axis
    :returns list: list of integral values
    """
    # ranges = [([sx[i], sy[j]], [sx[i+1], sy[j+1]]) for i in range(len(sx) - 1) for j in range(len(sy) - 1)]
    # corr = [mvn.mvnun(lower, upper, mu, S)[0] for lower, upper in ranges]
    # return corr

    # mean and covariance
    mu = np.array([0.0, 0.0])
    S = np.array([[1.0, rho], [rho, 1.0]])

    # callling mvn.mvnun is expensive, so we only calculate half of the matrix, then symmetrize
    # add half block, which is symmetric in x
    odd_odd = False
    ranges = [
        ([sx[i], sy[j]], [sx[i + 1], sy[j + 1]])
        for i in range((len(sx) - 1) // 2)
        for j in range(len(sy) - 1)
    ]
    # add odd middle row, which is symmetric in y
    if (len(sx) - 1) % 2 == 1:
        i = (len(sx) - 1) // 2
        ranges += [
            ([sx[i], sy[j]], [sx[i + 1], sy[j + 1]]) for j in range((len(sy) - 1) // 2)
        ]
        # add center point, add this only once
        if (len(sy) - 1) % 2 == 1:
            j = (len(sy) - 1) // 2
            ranges.append(([sx[i], sy[j]], [sx[i + 1], sy[j + 1]]))
            odd_odd = True

    corr = np.array([_calc_mvnun(lower, upper, mu, S) for lower, upper in ranges])
    # add second half, exclude center
    corr = np.concatenate([corr, corr if not odd_odd else corr[:-1]])
    return corr


def bivariate_normal_theory(
    rho: float,
    nx: int = -1,
    ny: int = -1,
    n: int = 1,
    sx: np.ndarray = None,
    sy: np.ndarray = None,
) -> np.ndarray:
    """Return binned pdf of bivariate normal distribution.

    This function returns a "perfect" binned bivariate normal distribution.

    :param float rho: tilt parameter
    :param int nx: number of uniform bins on x-axis. alternative to sx.
    :param int ny: number of uniform bins on y-axis. alternative to sy.
    :param np.ndarray sx: bin edges array of x-axis. default is None.
    :param np.ndarray sy: bin edges array of y-axis. default is None.
    :param int n: number of entries. default is one.
    :return: np.ndarray of binned bivariate normal pdf
    """

    if n < 1:
        raise ValueError("Number of entries needs to be one or greater.")
    if sx is None:
        sx = np.linspace(-5, 5, nx + 1)
    if sy is None:
        sy = np.linspace(-5, 5, ny + 1)

    bvn = np.zeros((ny, nx))
    for i in range(len(sx) - 1):
        for j in range(len(sy) - 1):
            lower = (sx[i], sy[j])
            upper = (sx[i + 1], sy[j + 1])
            p = _mvn_un(rho, lower, upper)
            bvn[j, i] = p
    bvn *= n

    # patch for entry levels that are below machine precision
    # (simulation does not work otherwise)
    bvn[bvn < np.finfo(np.float).eps] = np.finfo(np.float).eps

    return bvn


def chi2_from_phik(
    rho: float,
    n: int,
    subtract_from_chi2: float = 0,
    corr0: list = None,
    scale: float = None,
    sx: np.ndarray = None,
    sy: np.ndarray = None,
    pedestal: float = 0,
    nx: int = -1,
    ny: int = -1,
) -> float:
    """Calculate chi2-value of bivariate gauss having correlation value rho

    Calculate no-noise chi2 value of bivar gauss with correlation rho,
    with respect to bivariate gauss without any correlation.

    :param float rho: tilt parameter
    :param int n: number of records
    :param float subtract_from_chi2: value subtracted from chi2 calculation. default is 0.
    :param list corr0: mvn_array result for rho=0. Default is None.
    :param float scale: scale is multiplied with the chi2 if set.
    :param np.ndarray sx: bin edges array of x-axis. default is None.
    :param np.ndarray sy: bin edges array of y-axis. default is None.
    :param float pedestal: pedestal is added to the chi2 if set.
    :param int nx: number of uniform bins on x-axis. alternative to sx.
    :param int ny: number of uniform bins on y-axis. alternative to sy.
    :returns float: chi2 value
    """

    if sx is None:
        sx = np.linspace(-5, 5, nx + 1)

    if sy is None:
        sy = np.linspace(-5, 5, ny + 1)

    if corr0 is None:
        corr0 = _mvn_array(0, sx, sy)
    if scale is None:
        # scale ensures that for rho=1, chi2 is the maximum possible value
        corr1 = _mvn_array(1, sx, sy)
        delta_corr2 = (corr1 - corr0) ** 2
        # protect against division by zero
        ratio = np.divide(
            delta_corr2, corr0, out=np.zeros_like(delta_corr2), where=corr0 != 0
        )
        chi2_one = n * np.sum(ratio)
        # chi2_one = n * sum([((c1-c0)*(c1-c0)) / c0 for c0, c1 in zip(corr0, corr1)])
        chi2_max = n * min(nx - 1, ny - 1)
        scale = (chi2_max - pedestal) / chi2_one

    corrr = _mvn_array(rho, sx, sy)
    delta_corr2 = (corrr - corr0) ** 2
    # protect against division by zero
    ratio = np.divide(
        delta_corr2, corr0, out=np.zeros_like(delta_corr2), where=corr0 != 0
    )
    chi2_rho = n * np.sum(ratio)
    # chi2_rho = (n * sum([((cr-c0)*(cr-c0)) / c0 for c0, cr in zip(corr0, corrr)]))

    chi2 = pedestal + chi2_rho * scale
    return chi2 - subtract_from_chi2


def phik_from_chi2(
    chi2: float,
    n: int,
    nx: int,
    ny: int,
    sx: np.ndarray = None,
    sy: np.ndarray = None,
    pedestal: float = 0,
) -> float:
    """
    Correlation coefficient of bivariate gaussian derived from chi2-value

    Chi2-value gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records.
    Correlation coefficient value is between 0 and 1.

    Bivariate gaussian's range is set to [-5,5] by construction.

    :param float chi2: input chi2 value
    :param int n: number of records
    :param int nx: number of uniform bins on x-axis. alternative to sx.
    :param int ny: number of uniform bins on y-axis. alternative to sy.
    :param np.ndarray sx: bin edges array of x-axis. default is None.
    :param np.ndarray sy: bin edges array of y-axis. default is None.
    :param float pedestal: pedestal is added to the chi2 if set.
    :returns float: correlation coefficient
    """

    if pedestal < 0:
        raise ValueError("noise pedestal should be greater than zero.")

    if sx is None:
        sx = np.linspace(-5, 5, nx + 1)
    elif nx <= 1:
        raise ValueError("number of bins along x-axis is unknown")
    if sy is None:
        sy = np.linspace(-5, 5, ny + 1)
    elif ny <= 1:
        raise ValueError("number of bins along y-axis is unknown")

    corr0 = _mvn_array(0, sx, sy)

    # scale ensures that for rho=1, chi2 is the maximum possible value
    corr1 = _mvn_array(1, sx, sy)
    if 0 in corr0 and len(corr0) > 10000:
        warnings.warn(
            "Many cells: {0:d}. Are interval variables set correctly?".format(
                len(corr0)
            )
        )

    delta_corr2 = (corr1 - corr0) ** 2
    # protect against division by zero
    ratio = np.divide(
        delta_corr2, corr0, out=np.zeros_like(delta_corr2), where=corr0 != 0
    )
    chi2_one = n * np.sum(ratio)
    # chi2_one = n * sum([((c1-c0)*(c1-c0)) / c0 if c0 > 0 else 0 for c0,c1 in zip(corr0,corr1)])
    chi2_max = n * min(nx - 1, ny - 1)
    scale = (chi2_max - pedestal) / chi2_one
    if chi2 > chi2_max and np.isclose(chi2, chi2_max, atol=1e-14):
        chi2 = chi2_max

    # only solve for rho if chi2 exceeds noise pedestal
    if chi2 <= pedestal:
        return 0.0
    elif chi2 >= chi2_max:
        return 1.0

    rho = optimize.brentq(
        chi2_from_phik, 0, 1, args=(n, chi2, corr0, scale, sx, sy, pedestal), xtol=1e-5
    )
    return rho
