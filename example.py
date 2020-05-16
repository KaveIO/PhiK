import pandas as pd

import phik
from phik import resources, report

# open fake car insurance data
df = pd.read_csv( resources.fixture('fake_insurance_data.csv.gz') )
df.head()

# Pearson's correlation matrix between numeric variables (pandas functionality)
df.corr()

# get the phi_k correlation matrix between all variables
df.phik_matrix()

# get global correlations based on phi_k correlation matrix
df.global_phik()

# get the significance matrix (expressed as one-sided Z)
# of the hypothesis test of each variable-pair dependency
df.significance_matrix()

# contingency table of two columns
cols = ['mileage', 'car_size']
df[cols].hist2d()

# normalized residuals of contingency test applied to cols
df[cols].outlier_significance_matrix()

# show the normalized residuals of each variable-pair
df.outlier_significance_matrices()

# generate a phik correlation report and save as test.pdf
report.correlation_report(df, pdf_file_name='test.pdf')
