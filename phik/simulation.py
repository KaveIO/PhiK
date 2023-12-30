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

import numpy as np
from joblib import Parallel, delayed

from .statistics import get_dependent_frequency_estimates
from .statistics import get_chi2_using_dependent_frequency_estimates
from phik.simcore import CPP_SUPPORT, _sim_2d_data_patefield


NUMPY_INT_MAX = np.iinfo(np.int32).max - 1


def sim_2d_data(hist:np.ndarray, ndata:int=0) -> np.ndarray:
    """
    Simulate a 2 dimensional dataset given a 2 dimensional pdf

    :param array-like hist: contingency table, which contains the observed number of occurrences in each category.
        This table is used as probability density function.
    :param int ndata: number of simulations
    :return: simulated data
    """

    if ndata <= 0:
        ndata = int(np.rint(hist.sum()))
    if ndata <= 0:
        raise ValueError('ndata (or hist.sum()) has to be positive')

    # scale and ravel
    hc = hist[:] / hist.sum()
    hcr = hc.ravel()

    hout = np.random.multinomial(n=ndata, pvals=hcr)
    hout2d = np.reshape(hout, hc.shape)
    return hout2d


def sim_2d_data_patefield(data: np.ndarray, seed : int = None) -> np.ndarray:
    """
    Simulate a two dimensional dataset with fixed row and column totals.

    Simulation algorithm by Patefield:
    W. M. Patefield, Applied Statistics 30, 91 (1981)
    Python implementation inspired by (C version):
    https://people.sc.fsu.edu/~jburkardt/c_src/asa159/asa159.html

    :param data: contingency table, which contains the observed number of occurrences in each category.\
    :param seed: optional seed for the simulation, primarily for testing purposes.\
    This table is used as probability density function.
    :return: simulated data
    """

    if not CPP_SUPPORT:
        raise NotImplementedError(
            'Patefield requires a compiled extension that was not found.'
        )

    # number of rows and columns
    nrows, ncols = data.shape

    # totals per row and column
    # NOTE we assume that sum will fit in a 32 bit int
    nrowt = np.rint(data.sum(axis=1)).astype(np.int32)
    ncolt = np.rint(data.sum(axis=0)).astype(np.int32)

    # set seed if it is None
    seed = seed or np.random.randint(0, NUMPY_INT_MAX)

    # allocate memory that will be set by _sim_2d_data_patefield
    matrix = np.empty(nrows * ncols, dtype=np.int32)

    # simulate the data, returned through matrix inplace modification
    _sim_2d_data_patefield(nrows, ncols, nrowt, ncolt, seed, matrix)
    return matrix.reshape(ncols, nrows).T


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


def sim_chi2_distribution(values: np.ndarray, nsim:int=1000, lambda_:str='log-likelihood',
                          simulation_method:str='multinominal', alt_hypothesis:bool=False, njobs:int=-1) -> list:
    """
    Simulate 2D data and calculate the chi-square statistic for each simulated dataset.

    :param values: The contingency table. The table contains the observed number of occurrences in each category
    :param int nsim: number of simulations (optional, default=1000)
    :param str simulation_method: sampling method. Options: [multinominal, hypergeometric, row_product_multinominal,
        col_product_multinominal]
    :param str lambda_: test statistic. Available options are [pearson, log-likelihood].
    :param bool alt_hypothesis: if True, simulate values directly, and not its dependent frequency estimates.
    :param int njobs: number of parallel jobs used for simulation. default is -1. 1 uses no parallel jobs.
    :returns chi2s: list of chi2 values for each simulated dataset
    """
    exp_dep = get_dependent_frequency_estimates(values) if not alt_hypothesis else values

    if njobs == 1:
        chi2s = [_simulate_and_fit(exp_dep, simulation_method, lambda_) for _ in range(nsim)]
    else:
        chi2s = Parallel(n_jobs=njobs)(delayed(_simulate_and_fit)(exp_dep, simulation_method, lambda_)
                                        for _ in range(nsim))

    return chi2s


def _simulate_and_fit(exp_dep: np.ndarray, simulation_method: str='multinominal',
                      lambda_:str='log-likelihood') -> float:
    """split off simulate function to allow for parallellization"""
    simdata = sim_data(exp_dep, method=simulation_method)
    simchi2 = get_chi2_using_dependent_frequency_estimates(simdata, lambda_)
    return simchi2
