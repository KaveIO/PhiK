"""Project: PhiK - correlation analyzer library

Created: 2018/09/05

Description:
    Implementation of incomplete beta function

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""
import numpy as np
from scipy.special import gammaln
from typing import Union

def contfractbeta(a: float, b: float, x: float, ITMAX: int = 5000, EPS:float = 1.0e-7) -> float:
    """Continued fraction form of the incomplete Beta function.

    Code translated from: Numerical Recipes in C.

    Example kindly taken from blog:
    https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/ 

    :param float a: a
    :param float b: b
    :param float x: x
    :param int ITMAX: max number of iterations, default is 5000.
    :param float EPS: epsilon precision parameter, default is 1e-7.
    :returns: continued fraction form
    :rtype: float
    """
    az = 1.0
    bm = 1.0
    am = 1.0
    qab = a+b
    qap = a+1.0
    qam = a-1.0
    bz = 1.0-qab*x/qap
     
    for i in range(ITMAX+1):
        em = float(i+1)
        tem = em + em
        d = em*(b-em)*x/((qam+tem)*(a+tem))
        ap = az + d*am
        bp = bz+d*bm
        d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem))
        app = ap+d*az
        bpp = bp+d*bz
        aold = az
        am = ap/bpp
        bm = bp/bpp
        az = app/bpp
        bz = 1.0
        if abs(az-aold) < EPS*abs(az):
            return az
         
    raise ValueError('a={0:f} or b={1:f} too large, or ITMAX={2:d} too small to compute incomplete beta function.'.format(a,b,ITMAX))
    return 0


def incompbeta(a: float, b: float, x: float) -> float:
    '''Evaluation of incomplete beta function. 

    Code translated from: Numerical Recipes in C.

    Here a, b > 0 and 0 <= x <= 1. 
    This function requires contfractbeta(a,b,x, ITMAX = 200) 

    Example kindly taken from blog:
    https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/ 

    :param float a: a
    :param float b: b
    :param float x: x
    :returns: incomplete beta function
    :rtype: float
    ''' 
    # special cases
    if (x == 0):
        return 0;
    elif (x == 1):
        return 1;
    # default
    lbeta = gammaln(a+b) - gammaln(a) - gammaln(b) + a * np.log(x) + b * np.log(1-x)
    if (x < (a+1) / (a+b+2)):
        p = np.exp(lbeta) * contfractbeta(a, b, x) / a
    else:
        p = 1 - np.exp(lbeta) * contfractbeta(b, a, 1-x) / b
    return p


def log_incompbeta(a: float, b: float, x: float) -> Union[float,float]: 
    '''Evaluation of logarithm of incomplete beta function

    Logarithm of incomplete beta function is implemented to ensure sufficient precision
    for values very close to zero and one.

    Code translated from: Numerical Recipes in C.

    Here a, b > 0 and 0 <= x <= 1. 
    This function requires contfractbeta(a,b,x, ITMAX = 200) 

    Example kindly taken from blog:
    https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/ 

    :param float a: a
    :param float b: b
    :param float x: x
    :returns: tuple of log(incb) and log(1-incb)
    :rtype: tuple
    '''
    # special cases
    if (x == 0):
        return (-np.inf, 0)
    elif (x == 1):
        return (0, -np.inf)
    # default
    lbeta = gammaln(a+b) - gammaln(a) - gammaln(b) + a * np.log(x) + b * np.log(1-x)

    if (x < (a+1) / (a+b+2)):
        p = np.exp(lbeta) * contfractbeta(a, b, x) / a
        logp = lbeta + np.log(contfractbeta(a, b, x)) - np.log(a)
        logq = np.log(1-p)
    else:
        p = 1 - np.exp(lbeta) * ( contfractbeta(b, a, 1-x) / b )
        logp = np.log(p)
        logq = lbeta + np.log(contfractbeta(b, a, 1-x)) - np.log(b)
    return (logp, logq)
