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

from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .statistics import get_dependent_frequency_estimates
from .statistics import get_chi2_using_dependent_frequency_estimates

try:
    from numba import jit
except ImportError:
    def jit(func=None, **kwargs):
        if func:  # Called without other arguments, just return the function
            return func
        # Otherwise return a no-op decorator
        def decorator(func):
            return func
        return decorator


#@jit
def sim_2d_data(hist:np.ndarray, ndata:int=0) -> np.ndarray:
    """
    Simulate a 2 dimensional dataset given a 2 dimensional pdf

    :param array-like hist: contingency table, which contains the observed number of occurrences in each category. \
    This table is used as probability density function.
    :param int ndata: number of simulations
    :return: simulated data
    """

    if ndata <= 0:
        ndata = hist.sum()
    if ndata <= 0:
        raise ValueError('ndata (or hist.sum()) has to be positive')
    
    # scale and ravel
    hc = hist[:] / hist.sum()
    hcr = hc.ravel()

    hout = np.random.multinomial(n=ndata, pvals=hcr)
    hout2d = np.reshape(hout, hc.shape)
    return hout2d


# --- jit turned off for now, somehow not working for patefield; computer dependent!
#@jit
def sim_2d_data_patefield(data: np.ndarray) -> np.ndarray:
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

    # number of rows and columns
    nrows, ncols = data.shape

    # totals per row and column
    nrowt = data.sum(axis=1)
    ncolt = data.sum(axis=0)

    # total number of entries
    ntotal = int(np.round(ncolt.sum()))

    # Calculate log-factorials.
    x = 0.0
    fact = [x]

    for i in range(1, ntotal + 1):
        x += np.log(i)
        fact.append(x)

    # initialize matrix for end result
    matrix = np.empty(data.shape[0] * data.shape[1])
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
            done1 = False
            done2 = False

            while True:  # infinite loop!!!
                nlm = int(np.round((ia * id) / ie + 0.5))
                iap = int(np.round(ia + 1))
                idp = int(np.round(id + 1))
                igp = int(np.round(idp - nlm))
                ihp = int(np.round(iap - nlm))
                nlmp = int(np.round(nlm + 1))
                iip = int(np.round(ii + nlmp))

                x = np.exp(fact[iap - 1] + fact[ib] + fact[ic] + fact[idp - 1] -
                           fact[ie] - fact[nlmp - 1] - fact[igp - 1] - fact[ihp - 1] - fact[iip - 1])

                if (r < x) or np.isclose(r, x):
                    break

                # x, sumprb, y are float
                sumprb = x
                y = x
                nll = nlm
                lsp = False
                lsm = False

                # Increment entry in row L, column M.
                while not lsp:
                    j = int(np.round((id - nlm) * (ia - nlm)))

                    if np.isclose(j, 0):
                        # if j == 0:
                        lsp = True
                    else:
                        nlm += 1
                        x = x * j / (nlm * (ii + nlm))
                        sumprb += x

                        if (r < sumprb) or np.isclose(r, sumprb):
                            done1 = True
                            break

                    done2 = False

                    while not lsm:

                        # Decrement the entry in row L, column M.

                        j = nll * (ii + nll)

                        if np.isclose(j, 0):
                            lsm = True
                            break

                        nll -= 1
                        if np.isclose((id - nll) * (ia - nll), 0):  # make sure not to divide by zero
                            y = np.inf
                        else:
                            y = y * j / ((id - nll) * (ia - nll))
                        sumprb += y

                        if (r < sumprb) or np.isclose(r, sumprb):
                            nlm = nll
                            done2 = True
                            break
                    if done2:
                        break

                if done1 or done2:
                    break

                r = np.random.uniform()
                r = sumprb * r

            matrix[l + m * nrows] = nlm
            ia -= nlm
            jwork[m] -= nlm

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


def sim_2d_product_multinominal(data:np.ndarray, axis: int) -> np.ndarray:
    """
    Simulate 2 dimensional data with either row or column totals fixed.

    :param data: contingency table, which contains the observed number of occurrences in each category.\
    This table is used as probability density function.
    :param axis: fix row totals (0) or column totals (1).
    :return: simulated data
    """

    if axis == 1:
        return np.array([list(sim_2d_data(data[i])) for i in range(data.shape[0])])
    elif axis == 0:
        return np.array([list(sim_2d_data(data.T[i])) for i in range(data.shape[1])]).T
    else:
        raise NotImplementedError("Axis should be 0 (row) or 1 (column).")


@jit(forceobj=True)
def sim_data(data:np.ndarray, method:str='multinominal') -> np.ndarray:
    """
    Simulate a 2 dimensional dataset given a 2 dimensional pdf

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

    if method == 'multinominal':
        return sim_2d_data(data)
    elif method == 'hypergeometric':
        return sim_2d_data_patefield(data)
    elif method == 'row_product_multinominal':
        return sim_2d_product_multinominal(data, 0)
    elif method == 'col_product_multinominal':
        return sim_2d_product_multinominal(data, 1)
    else:
        raise NotImplementedError('selected method not recognized.')


# @jit
def sim_chi2_distribution(values: Union[pd.DataFrame, np.ndarray], nsim:int=1000, lambda_:str='log-likelihood',
                          simulation_method:str='multinominal', alt_hypothesis:bool=False) -> list:
    """
    Simulate 2D data and calculate the chi-square statistic for each simulated dataset.

    :param values: The contingency table. The table contains the observed number of occurrences in each category
    :param int nsim: number of simulations (optional, default=1000)
    :param str simulation_method: sampling method. Options: [multinominal, hypergeometric, row_product_multinominal,\
     col_product_multinominal]
    :param str lambda_: test statistic. Available options are [pearson, log-likelihood].
    :param bool alt_hypothesis: if True, simulate values directly, and not its dependent frequency estimates.
    :returns chi2s: list of chi2 values for each simulated dataset
    """
    values = values.values if isinstance(values, pd.DataFrame) else values

    exp_dep = get_dependent_frequency_estimates(values) if not alt_hypothesis else values

    from phik.config import ncores as NCORES
    chi2s = Parallel(n_jobs=NCORES)(delayed(_simulate_and_fit)(exp_dep, simulation_method, lambda_) for i in range(nsim))

    return chi2s


@jit(forceobj=True)
def _simulate_and_fit(exp_dep: np.ndarray, simulation_method: str='multinominal', lambda_:str='log-likelihood') -> float:
    """split off simulate function to allow for parallellization"""
    simdata = sim_data(exp_dep, method=simulation_method)
    simchi2 = get_chi2_using_dependent_frequency_estimates(simdata, lambda_)
    return simchi2
