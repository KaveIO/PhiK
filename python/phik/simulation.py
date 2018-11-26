"""Project: PhiK - correlation analyzer library

Created: 2018/09/05

Description:
    Helper functions to simulate 2D datasets

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import copy
import numpy as np
import pandas as pd
from numba import jit

from .statistics import get_dependent_frequency_estimates
from .statistics import get_chi2_using_dependent_frequency_estimates

@jit
def sim_2d_data(hist:np.ndarray, ndata:int=0) -> np.ndarray:
    """
    Simulate a 2 dimensional dataset given a 2 dimensional pdf

    :param array-like hist: contingency table, which contains the observed number of occurrences in each category. \
    This table is used as probability density function.
    :param int ndata: number of simulations
    :return: simulated data
    """
    if not isinstance(hist, np.ndarray):
        raise TypeError('hist is not a numpy array.')

    if ndata <= 0:
        ndata = hist.sum()
    assert ndata>0, 'ndata has to be positive.'
    
    # scale and ravel
    hc = copy.copy(hist) * ((1.0 * ndata) / hist.sum())
    hcr = hc.ravel() 

    # first estimate, unconstrained
    hout = np.zeros(hcr.shape)
    for i,h in enumerate(hcr):
        hout[i] = np.random.poisson(h)

    nbins = len(hcr)
    hmax = np.max(hcr)
    houtsum = int(np.sum(hout))

    # iterate until houtsum == ndata
    nextra = np.abs( int(ndata) - houtsum )
    wgt = -1 if houtsum>ndata else 1

    while nextra > 0:
        ibin = np.random.randint(0,nbins)
        h = hcr[ibin]
        hran = np.random.uniform(0,hmax)

        if hran<h:
            if wgt==1:
                hout[ibin] += 1
            else:
                if hout[ibin] > 0:
                    hout[ibin] -= 1
                else:
                    continue
            nextra -= 1

    # reshape
    hout2d = np.reshape(hout, hc.shape)
    return hout2d


# --- jit turned off for now, somehow not working for patefield; computer dependent!
#@jit
def sim_2d_data_patefield(data:np.ndarray) -> np.ndarray:
    """
    Simulate a two dimensional dataset with fixed row and column totals.

    Simulation algorithm by Patefield:
    W. M. Patefield, Applied Statistics 30, 91 (1981)
    Python implementation inspired by (C version):
    https://people.sc.fsu.edu/~jburkardt/c_src/asa159/asa159.html

    :param data: contingency table, which contains the observed number of occurrences in each category.\
    This table is used as probability density function.
    :return: simulated data
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data is not a numpy array.')

    # number of rows and columns
    nrows = data.shape[0]
    ncols = data.shape[1]

    # totals per row and column
    nrowt = data.sum(axis=1)
    ncolt = data.sum(axis=0)

    # total number of entries
    ntotal = int(np.round(ncolt.sum()))

    # Calculate log-factorials.
    x = 0.0
    fact = [x]

    for i in range(1, ntotal + 1):
        x = x + np.log(i)
        fact.append(x)

    # initialize matrix for end result
    matrix = np.zeros(data.shape[0] * data.shape[1])
    matrix[:] = np.nan

    # Construct a random matrix.

    jwork = []

    for i in range(0, ncols - 1):
        jwork.append(int(np.round(ncolt[i])))

    jc = ntotal

    for l in range(0, nrows - 1):
        nrowtl = int(np.round(nrowt[l]))
        ia = int(nrowtl)
        ic = int(jc)
        jc = int(jc - nrowtl)

        for m in range(0, ncols - 1):

            id = jwork[m]
            ie = int(np.round(ic))
            ic = int(np.round(ic - id))
            ib = int(np.round(ie - ia))
            ii = int(np.round(ib - id))

            # Test for zero entries in matrix.
            if np.isclose(ie, 0):
                ia = 0
                for j in range(m, ncols):
                    matrix[l + j * nrows] = 0
                break

            r = np.random.uniform()

            #  Compute the conditional expected value of MATRIX(L,M).

            done1 = 0

            while (True):  # infinit loop!!!
                nlm = int(np.round((ia * id) / (ie) + 0.5))
                iap = int(np.round(ia + 1))
                idp = int(np.round(id + 1))
                igp = int(np.round(idp - nlm))
                ihp = int(np.round(iap - nlm))
                nlmp = int(np.round(nlm + 1))
                iip = int(np.round(ii + nlmp))

                x = np.exp(fact[iap - 1] + fact[ib] + fact[ic] + fact[idp - 1] - \
                           fact[ie] - fact[nlmp - 1] - fact[igp - 1] - fact[ihp - 1] - fact[iip - 1])

                if (r < x) or np.isclose(r, x):
                    break

                # x, sumprb, y are float
                sumprb = x
                y = x
                nll = nlm
                lsp = 0
                lsm = 0

                # Increment entry in row L, column M.

                while not lsp:
                    j = int(np.round((id - nlm) * (ia - nlm)))

                    if np.isclose(j, 0):
                        # if j == 0:
                        lsp = 1
                    else:
                        nlm = nlm + 1
                        x = x * j / (nlm * (ii + nlm))
                        sumprb = sumprb + x

                        if (r < sumprb) or np.isclose(r, sumprb):
                            done1 = 1
                            break

                    done2 = 0

                    while not lsm:

                        # Decrement the entry in row L, column M.

                        j = nll * (ii + nll)

                        if np.isclose(j, 0):
                            lsm = 1
                            break

                        nll = nll - 1
                        if np.isclose((id - nll) * (ia - nll), 0):  # make sure not to divide by zero
                            y = np.inf
                        else:
                            y = y * j / ((id - nll) * (ia - nll))
                        sumprb = sumprb + y

                        if (r < sumprb) or np.isclose(r, sumprb):
                            nlm = nll
                            done2 = 1
                            break

                        if not lsp:
                            break

                    if done2:
                        break

                if done1:
                    break

                if done2:
                    break

                r = np.random.uniform()
                r = sumprb * r

            matrix[l + m * nrows] = nlm
            ia = ia - nlm
            jwork[m] = jwork[m] - nlm

        matrix[l + (ncols - 1) * nrows] = ia

    # Compute the last row.
    for j in range(ncols - 1):
        matrix[nrows - 1 + j * nrows] = jwork[j]

    # Compute last value at (m, l)
    matrix[nrows - 1 + (ncols - 1) * nrows] = int(np.round(ib - matrix[nrows - 1 + (ncols - 2) * nrows]))

    # reshape
    matrix = matrix.reshape(ncols, nrows).T

    # convert to int (note: int rounds everything down. All number are rounded properly before adding them to the matrix)
    matrix = matrix.astype(int)

    return matrix


def sim_2d_product_multinominal(data:np.ndarray, axis:str) -> np.ndarray:
    """
    Simulate 2 dimensional data with either row or column totals fixed.

    :param data: contingency table, which contains the observed number of occurrences in each category.\
    This table is used as probability density function.
    :param axis: fix row totals (rows) or column totals (cols).
    :return: simulated data
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data is not a numpy array.')

    if axis == 'cols':
        return np.array([list(sim_2d_data(data[i])) for i in range(data.shape[0])])
    if axis == 'rows':
        return np.array([list(sim_2d_data(data.T[i])) for i in range(data.shape[1])]).T
    else:
        raise ValueError


@jit
def sim_data(data:np.ndarray, method:str='multinominal') -> np.ndarray:
    """
    Simulate a 2 dimenstional dataset given a 2 dimensional pdf

    Several simulation methods are provided:

     - multinominal: Only the total number of records is fixed.
     - row_product_multinominal: The row totals fixed in the sampling.
     - col_product_multinominal: The column totals fixed in the sampling.
     - hypergeometric: Both the row or column totals are fixed in the sampling. Note that this type of sampling is\
    only available when row and column totals are integers.

    :param data: contingency table
    :param str method: sampling method. Options: [multinominal, hypergeometric, row_product_multinominal,\
     col_product_multinominal]
    :return: simulated data
    """
    assert method in ['multinominal', 'hypergeometric', 'row_product_multinominal', 'col_product_multinominal'], 'selected method not recognized.'
    if not isinstance(data, np.ndarray):
        raise TypeError('data is not a numpy array.')

    if method == 'multinominal':
        return sim_2d_data(data)
    elif method == 'hypergeometric':
        return sim_2d_data_patefield(data)
    elif method == 'row_product_multinominal':
        return sim_2d_product_multinominal(data, 'rows')
    elif method == 'col_product_multinominal':
        return sim_2d_product_multinominal(data, 'cols')
    else:
        raise ValueError


@jit
def sim_chi2_distribution(values, nsim:int=1000, lambda_:str='log-likelihood', simulation_method:str='multinominal') -> list:
    """
    Simulate 2D data and calculate the chi-square statistic for each simulated dataset.

    :param values: The contingency table. The table contains the observed number of occurrences in each category
    :param int nsim: number of simulations (optional, default=1000)
    :param str simulation_method: sampling method. Options: [multinominal, hypergeometric, row_product_multinominal,\
     col_product_multinominal]
    :param str lambda_: test statistic. Available options are [pearson, log-likelihood].
    :returns chi2s: list of chi2 values for each simulated dataset
    """
    vals = values.values if isinstance(values, pd.DataFrame) else values
    if not isinstance(vals, np.ndarray):
        raise TypeError('values is not a numpy array.')

    exp_dep = get_dependent_frequency_estimates(values)

    chi2s = []

    for i in range(nsim):
        simdata = sim_data(exp_dep, method=simulation_method)
        simchi2 = get_chi2_using_dependent_frequency_estimates(simdata, lambda_)
        chi2s.append(simchi2)

    return chi2s
