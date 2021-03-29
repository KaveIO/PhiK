"""Project: PhiK - correlation analyzer library

Module: phik.decorators.pandas

Created: 2018/11/14

Description:
    Decorators for pandas DataFrame objects

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

from pandas import DataFrame, Series

# add function to create a 2d histogram
from phik.binning import hist2d, hist2d_from_array
DataFrame.hist2d = hist2d
Series.hist2d = hist2d_from_array

# add phik correlation matrix function
from phik.phik import phik_matrix, global_phik_array
DataFrame.phik_matrix = phik_matrix
DataFrame.global_phik = global_phik_array

# add significance matrix function for variable dependencies
from phik.significance import significance_matrix
DataFrame.significance_matrix = significance_matrix

# outlier matrix
from phik.outliers import outlier_significance_matrices, outlier_significance_matrix, outlier_significance_from_array
DataFrame.outlier_significance_matrices = outlier_significance_matrices
DataFrame.outlier_significance_matrix = outlier_significance_matrix
Series.outlier_significance_matrix = outlier_significance_from_array
