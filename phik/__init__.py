# flake8: noqa
from phik.version import version as __version__

# pandas dataframe decorators
from phik import decorators

# array functions
from phik.phik import phik_from_array
from phik.significance import significance_from_array
from phik.outliers import outlier_significance_from_array

# dataframe functions
from phik.phik import phik_matrix, global_phik_array
from phik.significance import significance_matrix
from phik.outliers import outlier_significance_matrices, outlier_significance_matrix
