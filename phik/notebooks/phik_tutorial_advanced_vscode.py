# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Phi_K advanced tutorial
# 
# This notebook guides you through the more advanced functionality of the phik package. This notebook will not cover all the underlying theory, but will just attempt to give an overview of all the options that are available. For a theoretical description the user is referred to our paper.
# 
# The package offers functionality on three related topics:
# 
# 1. Phik correlation matrix
# 2. Significance matrix
# 3. Outlier significance matrix

# %%
# get_ipython().run_cell_magic('capture', '', '# install phik (if not installed yet)\nimport sys\n\n!"{sys.executable}" -m pip install phik')


# %%
# import standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import phik

from phik import resources
from phik.binning import bin_data
from phik.decorators import *
from phik.report import plot_correlation_matrix

# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
# if one changes something in the phik-package one can automatically reload the package or module
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# # Load data
# 
# A simulated dataset is part of the phik-package. The dataset concerns car insurance data. Load the dataset here:

# %%
data = pd.read_csv( resources.fixture('fake_insurance_data.csv.gz') )
# %%
data.head()

# %% [markdown]
# ## Specify bin types
# 
# The phik-package offers a way to calculate correlations between variables of mixed types. Variable types can be inferred automatically although we recommend to variable types to be specified by the user. 
# 
# Because interval type variables need to be binned in order to calculate phik and the significance, a list of interval variables is created.

# %%
data_types = {'severity': 'interval',
             'driver_age':'interval',
             'satisfaction':'ordinal',
             'mileage':'interval',
             'car_size':'ordinal',
             'car_use':'ordinal',
             'car_color':'categorical',
             'area':'categorical'}

interval_cols = [col for col, v in data_types.items() if v=='interval' and col in data.columns]
interval_cols
# interval_cols is used below

# %% [markdown]
# # Phik correlation matrix
# 
# Now let's start calculating the correlation phik between pairs of variables. 
# 
# Note that the original dataset is used as input, the binning of interval variables is done automatically.

# %%
phik_overview = data.phik_matrix(interval_cols=interval_cols)
phik_overview

# %% [markdown]
# ### Specify binning per interval variable
# 
# Binning can be set per interval variable individually. One can set the number of bins, or specify a list of bin edges. Note that the measured phik correlation is dependent on the chosen binning. 
# The default binning is uniform between the min and max values of the interval variable.

# %%
bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
phik_overview = data.phik_matrix(interval_cols=interval_cols, bins=bins)
phik_overview

# %% [markdown]
# ### Do not apply noise correction
# 
# For low statistics samples often a correlation larger than zero is measured when no correlation is actually present in the true underlying distribution. This is not only the case for phik, but also for the pearson correlation and Cramer's phi (see figure 4 in <font color='red'> XX </font>). In the phik calculation a noise correction is applied by default, to take into account erroneous correlation values as a result of low statistics. To switch off this noise cancellation (not recommended), do:

# %%
phik_overview = data.phik_matrix(interval_cols=interval_cols, noise_correction=False)
phik_overview

# %% [markdown]
# ### Using a different expectation histogram
# 
# By default phik compares the 2d distribution of two (binned) variables with the distribution that assumes no dependency between them. One can also change the expected distribution though. Phi_K is calculated in the same way, but using the other expectation distribution. 

# %%
from phik.binning import auto_bin_data
from phik.phik import phik_observed_vs_expected_from_rebinned_df, phik_from_hist2d
from phik.statistics import get_dependent_frequency_estimates


# %%
# get observed 2d histogram of two variables
cols = ["mileage", "car_size"]
icols = ["mileage"]
observed = data[cols].hist2d(interval_cols=icols).values

# %%
observed
# %%
cols
# %%
data[cols].hist2d(interval_cols=icols).values
# %%
# default phik evaluation from observed distribution
phik_value = phik_from_hist2d(observed)
print (phik_value)


# %%
# phik evaluation from an observed and expected distribution
expected = get_dependent_frequency_estimates(observed)
phik_value = phik_from_hist2d(observed=observed, expected=expected)
print (phik_value)


# %%
# one can also compare two datasets against each other, and get a full phik matrix that way.
# this needs binned datasets though. 
# (the user needs to make sure the binnings of both datasets are identical.) 
data_binned, _ = auto_bin_data(data, interval_cols=interval_cols)


# %%
# here we are comparing data_binned against itself
phik_matrix = phik_observed_vs_expected_from_rebinned_df(data_binned, data_binned)


# %%
# all off-diagonal entries are zero, meaning the all 2d distributions of both datasets are identical.
# (by construction the diagonal is one.)
phik_matrix

# %% [markdown]
# # Statistical significance of the correlation
# 
# When assessing correlations it is good practise to evaluate both the correlation and the significance of the correlation: a large correlation may be statistically insignificant, and vice versa a small correlation may be very significant. For instance, scipy.stats.pearsonr returns both the pearson correlation and the p-value. Similarly, the phik package offers functionality the calculate a significance matrix. Significance is defined as:
# 
# $$Z = \Phi^{-1}(1-p)\ ;\quad \Phi(z)=\frac{1}{\sqrt{2\pi}} \int_{-\infty}^{z} e^{-t^{2}/2}\,{\rm d}t $$
# 
# Several corrections to the 'standard' p-value calculation are taken into account, making the method more robust for low statistics and sparse data cases. The user is referred to our paper for more details.
# 
# Due to the corrections, the significance calculation can take a few seconds.

# %%
significance_overview = data.significance_matrix(interval_cols=interval_cols)
significance_overview

# %% [markdown]
# ### Specify binning per interval variable
# Binning can be set per interval variable individually. One can set the number of bins, or specify a list of bin edges. Note that the measure phik correlation is dependent on the chosen binning.

# %%
bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
significance_overview = data.significance_matrix(interval_cols=interval_cols, bins=bins)
significance_overview

# %% [markdown]
# ### Specify significance method
# 
# The recommended method to calculate the significance of the correlation is a hybrid approach, which uses the G-test statistic. The number of degrees of freedom and an analytical, empirical description of the $\chi^2$ distribution are sed, based on Monte Carlo simulations. This method works well for both high as low statistics samples.
# 
# Other approaches to calculate the significance are implemented:
# - asymptotic: fast, but over-estimates the number of degrees of freedom for low statistics samples, leading to erroneous values of the significance
# - MC: Many simulated samples are needed to accurately measure significances larger than 3, making this method computationally expensive.
# 

# %%
significance_overview = data.significance_matrix(interval_cols=interval_cols, significance_method='asymptotic')
significance_overview

# %% 
significance_overview = data.significance_matrix(interval_cols=interval_cols, significance_method='hybrid')
significance_overview

# %% [markdown]
# ### Simulation method
# 
# The chi2 of a contingency table is measured using a comparison of the expected frequencies with the true frequencies in a contingency table. The expected frequencies can be simulated in a variety of ways. The following methods are implemented:
# 
#  - multinominal: Only the total number of records is fixed. (default)
#  - row_product_multinominal: The row totals fixed in the sampling.
#  - col_product_multinominal: The column totals fixed in the sampling.
#  - hypergeometric: Both the row or column totals are fixed in the sampling. (Note that this type of sampling is only available when row and column totals are integers, which is usually the case.)

# %%
# --- Warning, can be slow
#     turned off here by default for unit testing purposes

significance_overview = data.significance_matrix(interval_cols=interval_cols, simulation_method='hypergeometric')
significance_overview

# %% [markdown]
# ### Expected frequencies

# %%
from phik.simulation import sim_2d_data_patefield, sim_2d_product_multinominal, sim_2d_data


# %%
inputdata = data[['driver_age', 'area']].hist2d(interval_cols=['driver_age'])
inputdata

# %% [markdown]
# #### Multinominal

# %%
simdata = sim_2d_data(inputdata.values)
print('data total:', inputdata.sum().sum())
print('sim  total:', simdata.sum().sum())
print('data row totals:', inputdata.sum(axis=0).values)
print('sim  row totals:', simdata.sum(axis=0))
print('data column totals:', inputdata.sum(axis=1).values)
print('sim  column totals:', simdata.sum(axis=1))
# %%
simdata
 # %%
 
# %% [markdown]
# #### product multinominal

# %%
simdata = sim_2d_product_multinominal(inputdata.values, axis=0)
print('data total:', inputdata.sum().sum())
print('sim  total:', simdata.sum().sum())
print('data row totals:', inputdata.sum(axis=0).astype(int).values)
print('sim  row totals:', simdata.sum(axis=0).astype(int))
print('data column totals:', inputdata.sum(axis=1).astype(int).values)
print('sim  column totals:', simdata.sum(axis=1).astype(int))

# %% [markdown]
# #### hypergeometric ("patefield")

# %%
# patefield simulation needs compiled c++ code.
# only run this if the python binding to the (compiled) patefiled simulation function is found.
try:
    from phik.simcore import _sim_2d_data_patefield
    CPP_SUPPORT = True
except ImportError:
    CPP_SUPPORT = False

if CPP_SUPPORT:
    simdata = sim_2d_data_patefield(inputdata.values)
    print('data total:', inputdata.sum().sum())
    print('sim  total:', simdata.sum().sum())
    print('data row totals:', inputdata.sum(axis=0).astype(int).values)
    print('sim  row totals:', simdata.sum(axis=0))
    print('data column totals:', inputdata.sum(axis=1).astype(int).values)
    print('sim  column totals:', simdata.sum(axis=1))

# %% [markdown]
# # Outlier significance
# 
# The normal pearson correlation between two interval variables is easy to interpret. However, the phik correlation between two variables of mixed type is not always easy to interpret, especially when it concerns categorical variables. Therefore, functionality is provided to detect "outliers": excesses and deficits over the expected frequencies  in the contingency table of two variables. 
# 
# %% [markdown]
# ### Example 1: mileage versus car_size
# %% [markdown]
# For the categorical variable pair mileage - car_size we measured:
# 
# $$\phi_k = 0.77 \, ,\quad\quad \mathrm{significance} = 46.3$$
# 
# Let's use the outlier significance functionality to gain a better understanding of this significance correlation between mileage and car size.
# 

# %%
c0 = 'mileage'
c1 = 'car_size'

tmp_interval_cols = ['mileage']


# %%
outlier_signifs, binning_dict = data[[c0,c1]].outlier_significance_matrix(interval_cols=tmp_interval_cols, 
                                                                          retbins=True)
outlier_signifs

# %% [markdown]
# ### Specify binning per interval variable
# Binning can be set per interval variable individually. One can set the number of bins, or specify a list of bin edges. 
# 
# Note: in case a bin is created without any records this bin will be automatically dropped in the phik and (outlier) significance calculations. However, in the outlier significance calculation this will currently lead to an error as the number of provided bin edges does not match the number of bins anymore.

# %%
bins = [0,1E2, 1E3, 1E4, 1E5, 1E6]
outlier_signifs, binning_dict = data[[c0,c1]].outlier_significance_matrix(interval_cols=tmp_interval_cols, 
                                                                          bins=bins, retbins=True)
outlier_signifs

# %% [markdown]
# ### Specify binning per interval variable -- dealing with underflow and overflow
# 
# When specifying custom bins as situation can occur when the minimal (maximum) value in the data is smaller (larger) than the minimum (maximum) bin edge. Data points outside the specified range will be collected in the underflow (UF) and overflow (OF) bins. One can choose how to deal with these under/overflow bins, by setting the drop_underflow and drop_overflow variables.
# 
# Note that the drop_underflow and drop_overflow options are also available for the calculation of the phik matrix and the significance matrix.

# %%
bins = [1E2, 1E3, 1E4, 1E5]
outlier_signifs, binning_dict = data[[c0,c1]].outlier_significance_matrix(interval_cols=tmp_interval_cols, 
                                                                          bins=bins, retbins=True, 
                                                                          drop_underflow=False,
                                                                          drop_overflow=False)
outlier_signifs

# %% [markdown]
# ### Dealing with NaN's in the data
# %% [markdown]
# Let's add some missing values to our data

# %%
data.loc[np.random.choice(range(len(data)), size=10), 'car_size'] = np.nan
data.loc[np.random.choice(range(len(data)), size=10), 'mileage'] = np.nan

# %% [markdown]
# Sometimes there can be information in the missing values and in which case you might want to consider the NaN values as a separate category. This can be achieved by setting the dropna argument to False.

# %%
bins = [1E2, 1E3, 1E4, 1E5]
outlier_signifs, binning_dict = data[[c0,c1]].outlier_significance_matrix(interval_cols=tmp_interval_cols, 
                                                                          bins=bins, retbins=True, 
                                                                          drop_underflow=False,
                                                                          drop_overflow=False,
                                                                          dropna=False)
outlier_signifs

# %% [markdown]
# Here OF and UF are the underflow and overflow bin of car_size, respectively.
# 
# To just ignore records with missing values set dropna to True (default).

# %%
bins = [1E2, 1E3, 1E4, 1E5]
outlier_signifs, binning_dict = data[[c0,c1]].outlier_significance_matrix(interval_cols=tmp_interval_cols, 
                                                                          bins=bins, retbins=True, 
                                                                          drop_underflow=False,
                                                                          drop_overflow=False,
                                                                          dropna=True)
outlier_signifs

# %% [markdown]
# Note that the dropna option is also available for the calculation of the phik matrix and the significance matrix.

