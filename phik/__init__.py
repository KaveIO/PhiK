# flake8: noqa
import importlib.metadata

from phik import decorators
from phik.outliers import (
    outlier_significance_from_array,
    outlier_significance_matrices,
    outlier_significance_matrix,
)
from phik.phik import global_phik_array, phik_from_array, phik_matrix
from phik.significance import significance_from_array, significance_matrix

__version__ = importlib.metadata.version("phik")

__all__ = [
    "decorators",
    "phik_from_array",
    "significance_from_array",
    "outlier_significance_from_array",
    "phik_matrix",
    "global_phik_array",
    "significance_matrix",
    "outlier_significance_matrices",
    "outlier_significance_matrix",
]
