"""Project: PhiK - correlation analyzer library

Created: 2019/11/23

Description:
    Convert Pearson correlation value into a chi2 value of a contingency test 
    matrix of a bivariate gaussion, and vice-versa. 
    Calculation uses scipy's mvn library.

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import numpy as np
from scipy.stats import mvn
from scipy import optimize


def _mvn_un(rho: float, lower: tuple, upper: tuple) -> float:
    '''Perform integral of bivariate normal gauss with correlation

    Integral is performed using scipy's mvn library.

    :param float rho: tilt parameter
    :param tuple lower: tuple of lower corner of integral area
    :param tuple upper: tuple of upper corner of integral area
    :returns float: integral value
    '''
    mu = np.array([0., 0.])
    S = np.array([[1.,rho],[rho,1.0]])
    p,i = mvn.mvnun(lower,upper,mu,S)
    return p


def _mvn_array(rho: float, sx: np.ndarray, sy: np.ndarray) -> list:
    '''Array of integrals over bivariate normal gauss with correlation

    Integrals are performed using scipy's mvn library.
    
    :param float rho: tilt parameter
    :param np.ndarray sx: bin edges array of x-axis
    :param np.ndarray sy: bin edges array of y-axis
    :returns list: list of integral values
    '''
    corr = []
    for i in range(len(sx)-1):
        for j in range(len(sy)-1):
            lower = [sx[i],sy[j]]
            upper = [sx[i+1],sy[j+1]]
            p = _mvn_un(rho,lower,upper)
            corr.append(p)
    return corr


def chi2_from_phik(rho: float, n: int, subtract_from_chi2:float=0,
                   corr0:list=None, scale:float=None, sx:np.ndarray=None, sy:np.ndarray=None,
                   pedestal:float=0, nx:int=-1, ny:int=-1) -> float:
    '''Calculate chi2-value of bivariate gauss having correlation value rho
    
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
    '''
    assert nx>1 or sx is not None, 'number of bins along x-axis is unknown'
    assert ny>1 or sy is not None, 'number of bins along y-axis is unknown'
    if sx is None:
        sx = np.linspace(-5,5,nx+1)
    if sy is None:
        sy = np.linspace(-5,5,ny+1)
    if corr0 is None:
        corr0 = _mvn_array(0, sx, sy)
    if scale is None:
        # scale ensures that for rho=1, chi2 is the maximum possible value
        corr1 = _mvn_array(1, sx, sy)
        chi2_one = n * sum([((c1-c0)*(c1-c0)) / c0 for c0,c1 in zip(corr0,corr1)])
        chi2_max = n * min(nx-1, ny-1)
        scale = (chi2_max - pedestal) / chi2_one

    corrr = _mvn_array(rho, sx, sy)
    chi2 = pedestal + ( n * sum([((cr-c0)*(cr-c0)) / c0 for c0,cr in zip(corr0,corrr)]) ) * scale
    return chi2 - subtract_from_chi2


def phik_from_chi2(chi2:float, n:int, nx:int, ny:int, sx:np.ndarray=None, sy:np.ndarray=None, pedestal:float=0) -> float:
    '''
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
    '''
    assert nx>1 or sx is not None, 'number of bins along x-axis is unknown'
    assert ny>1 or sy is not None, 'number of bins along y-axis is unknown'
    assert pedestal>=0, 'noise pedestal should be greater than zero.'
    if sx is None:
        sx = np.linspace(-5,5,nx+1)
    if sy is None:
        sy = np.linspace(-5,5,ny+1)
    corr0 = _mvn_array(0, sx, sy)

    # scale ensures that for rho=1, chi2 is the maximum possible value
    corr1 = _mvn_array(1, sx, sy)
    chi2_one = n * sum([((c1-c0)*(c1-c0)) / c0 for c0,c1 in zip(corr0,corr1)])
    chi2_max = n * min(nx-1, ny-1)
    scale = (chi2_max - pedestal) / chi2_one

    # only solve for rho if chi2 exceeds noise pedestal
    if chi2 <= pedestal:
        return 0

    rho = optimize.brentq(chi2_from_phik, 0, 1, args=(n, chi2, corr0, scale, sx, sy, pedestal))
    return rho
