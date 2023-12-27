# # Phi_K basic tutorial
#
# This notebook guides you through the basic functionality of the phik package. The package offers functionality on three related topics:
#
# 1. Phik correlation matrix
# 2. Significance matrix
# 3. Outlier significance matrix
#
# For more information on the underlying theory, the user is referred to our paper.

import itertools

import matplotlib.pyplot as plt
# +
# import standard packages
import numpy as np
import pandas as pd

import phik
from phik import resources
from phik.binning import bin_data
from phik.report import plot_correlation_matrix

# # Load data
#
# A simulated dataset is part of the phik-package. The dataset concerns fake car insurance data. Load the dataset here:


def test_basic_notebook():
    data = pd.read_csv(resources.fixture("fake_insurance_data.csv.gz"))

    # # Take a first look at the data

    # Let's use a simple data.head() to get an idea of what the data looks like and inspect the different types of variables.

    data.head()

    # # Specify bin types
    #
    # The phik-package offers a way to calculate correlations between variables of mixed types. Variable types can be inferred automatically although we recommend variable types to be specified by the user.
    #
    # Because interval type variables need to be binned in order to calculate phik and the significance, a list of interval variables is created.

    # +
    data_types = {
        "severity": "interval",
        "driver_age": "interval",
        "satisfaction": "ordinal",
        "mileage": "interval",
        "car_size": "ordinal",
        "car_use": "ordinal",
        "car_color": "categorical",
        "area": "categorical",
    }

    interval_cols = [
        col for col, v in data_types.items() if v == "interval" and col in data.columns
    ]
    # -

    # # Visually inspect pairwise correlations

    # ## Bin the interval variables
    #
    # To get a feeling for the data, let's bin the interval variables and create 2d histograms to inspect the correlations between variables. By binning the interval variables we can treat all variable types in the same way.
    #

    # bin the interval variables
    data_binned, binning_dict = bin_data(data, cols=interval_cols, retbins=True)

    # +
    # plot each variable pair
    plt.rc("text", usetex=False)

    n = 0
    for i in range(len(data.columns)):
        n = n + i

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    ndecimals = 0

    for i, comb in enumerate(itertools.combinations(data_binned.columns.values, 2)):
        c = int(i % ncols)
        r = int((i - c) / ncols)

        # get data
        c0, c1 = comb
        datahist = (
            data_binned.groupby([c0, c1])[c0].count().to_frame().unstack().fillna(0)
        )
        datahist.columns = datahist.columns.droplevel()

        # plot data
        img = axes[r][c].pcolormesh(datahist.values, edgecolor="w", linewidth=1)

        # axis ticks and tick labels
        if c0 in binning_dict.keys():
            ylabels = [
                "{1:.{0}f}_{2:.{0}f}".format(
                    ndecimals, binning_dict[c0][i][0], binning_dict[c0][i][1]
                )
                for i in range(len(binning_dict[c0]))
            ]
        else:
            ylabels = datahist.index

        if c1 in binning_dict.keys():
            xlabels = [
                "{1:.{0}f}_{2:.{0}f}".format(
                    ndecimals, binning_dict[c1][i][0], binning_dict[c1][i][1]
                )
                for i in range(len(binning_dict[c1]))
            ]
        else:
            xlabels = datahist.columns

        # axis labels
        axes[r][c].set_yticks(np.arange(len(ylabels)) + 0.5)
        axes[r][c].set_xticks(np.arange(len(xlabels)) + 0.5)
        axes[r][c].set_xticklabels(xlabels, rotation="vertical")
        axes[r][c].set_yticklabels(ylabels, rotation="horizontal")
        axes[r][c].set_xlabel(datahist.columns.name)
        axes[r][c].set_ylabel(datahist.index.name)
        axes[r][c].set_title("data")

    plt.tight_layout()

    # -

    # # Correlation: mileage vs car_size
    #
    # From the above plots it seems like there might be an interesting a correlation between mileage and car_size. Let's see what phik correlation is measured for this data.

    # +
    x, y = data[["mileage", "car_size"]].T.values

    print("phik         =  %.2f" % phik.phik_from_array(x, y, num_vars=["x"]))
    print("significance = %.2f" % phik.significance_from_array(x, y, num_vars=["x"])[1])

    # -

    # Indeed there is a correlation between these variables and the correlation is also significant. To better understand the correlation, we can have a look at the significance of excesses and deficits in the 2-dimensional contingency table, so-called "outlier significances".

    phik.outlier_significance_from_array(x, y, num_vars=["x"])

    # The values displayed in the matrix are the significances of the outlier frequencies, i.e. a large value means that the measured frequency for that bin is significantly different from the expected frequency in that bin.
    #
    # Let's visualise for easier interpretation.

    # +
    outlier_signifs = phik.outlier_significance_from_array(x, y, num_vars=["x"])

    zvalues = outlier_signifs.values
    xlabels = outlier_signifs.columns
    ylabels = outlier_signifs.index
    xlabel = "x"
    ylabel = "y"

    plot_correlation_matrix(
        zvalues,
        x_labels=xlabels,
        y_labels=ylabels,
        x_label=xlabel,
        y_label=ylabel,
        vmin=-5,
        vmax=5,
        title="outlier significance",
        identity_layout=False,
        fontsize_factor=1.2,
    )
    # -

    # # $\phi_k$ functions for dataframes
    #
    # In our data we have 5 different columns, meaning we have to evaluate 4+3+2+1=10 pairs of variables for possible correlations. In a large dataset, with many different variables, this can easily become a cumbersome task. Can we do this more efficient? yes! We have provided functions that work on dataframes, to allow you to calculate the phik correlation, significance and outlier significance for all different variable combinations at once.
    #

    # The functions are by default available after import of the phik package.

    # # $\phi_k$ correlation matrix
    #
    # Now let's start calculating the phik correlation coefficient between pairs of variables.
    #
    # Note that the original dataset is used as input, the binning of interval variables is done automatically.

    phik_overview = data.phik_matrix(interval_cols=interval_cols)
    phik_overview

    # When no interval columns are provided, the code makes an educated guess

    data.phik_matrix()

    plot_correlation_matrix(
        phik_overview.values,
        x_labels=phik_overview.columns,
        y_labels=phik_overview.index,
        vmin=0,
        vmax=1,
        color_map="Blues",
        title=r"correlation $\phi_K$",
        fontsize_factor=1.5,
        figsize=(7, 5.5),
    )
    plt.tight_layout()

    # # Global correlation: $g_k$
    #
    # The global correlation coefficient is a measure of the total correlation of one variable to all other variables in the dataset. They give an indication of how well on variable can be modelled in terms of the other variables. A calculation of the global correlation coefficient is provided within the phik package.

    global_correlation, global_labels = data.global_phik(interval_cols=interval_cols)
    for c, l in zip(global_correlation, global_labels):
        print(l, c[0])

    plot_correlation_matrix(
        global_correlation,
        x_labels=[""],
        y_labels=global_labels,
        vmin=0,
        vmax=1,
        figsize=(3.5, 4),
        color_map="Blues",
        title=r"$g_k$",
        fontsize_factor=1.5,
    )
    plt.tight_layout()

    # # Statistical significance of the correlation: $Z$-score
    #
    # When assessing correlations it is good practise to evaluate both the correlation and the significance of the correlation: a large correlation may be statistically insignificant, and vice versa a small correlation may be very significant. For instance, scipy.stats.pearsonr returns both the pearson correlation and the p-value. Similarly, the phik package offers functionality the calculate a significance matrix. Significance is defined as:
    #
    # $$Z = \Phi^{-1}(1-p)\ ;\quad \Phi(z)=\frac{1}{\sqrt{2\pi}} \int_{-\infty}^{z} e^{-t^{2}/2}\,{\rm d}t $$
    #
    # Several corrections to the 'standard' p-value calculation are taken into account, making the method more robust for low statistics and sparse data cases. The user is referred to our paper for more details.
    #
    # As a result, the calculation may take a few seconds.

    significance_overview = data.significance_matrix(interval_cols=interval_cols)
    significance_overview

    plot_correlation_matrix(
        significance_overview.fillna(0).values,
        x_labels=significance_overview.columns,
        y_labels=significance_overview.index,
        vmin=-5,
        vmax=5,
        title="significance",
        usetex=False,
        fontsize_factor=1.5,
        figsize=(7, 5.5),
    )
    plt.tight_layout()

    # # Outlier significance
    #
    # The normal pearson correlation between two interval variables is easy to interpret. However, the phik correlation between two variables of mixed type is not always easy to interpret, especially when it concerns categorical variables. Therefore, functionality is provided to detect "outliers": excesses and deficits over the expected frequencies in the contingency table of two variables.
    #

    # ### Example 1: car_color versus area
    #
    # For the categorical variable pair car_color - area we measured:
    #
    # $$\phi_k = 0.59 \, ,\quad\quad \mathrm{significance} = 37.6$$
    #
    # Let's use the outlier significance functionality to gain a better understanding of the significance correlation between car color and area.
    #

    c1 = "car_color"
    c0 = "area"

    outlier_signifs, binning_dict = data[[c0, c1]].outlier_significance_matrix(
        retbins=True
    )
    outlier_signifs

    # +
    zvalues = outlier_signifs.values
    xlabels = binning_dict[c1] if c1 in binning_dict.keys() else outlier_signifs.columns
    ylabels = binning_dict[c0] if c0 in binning_dict.keys() else outlier_signifs.index
    xlabel = c1
    ylabel = c0

    plot_correlation_matrix(
        zvalues,
        x_labels=xlabels,
        y_labels=ylabels,
        x_label=xlabel,
        y_label=ylabel,
        vmin=-5,
        vmax=5,
        title="outlier significance",
        identity_layout=False,
        fontsize_factor=1.2,
    )
    # -

    # The significance of each cell is expressed in terms of Z (one-sided).
    #
    # Interesting, owners of a green car are more likely to live in the country side, and black cars are more likely to travel on unpaved roads!

    # ### Example 2: mileage versus car_size

    # For the categorical variable pair mileage - car_size we measured:
    #
    # $$\phi_k = 0.77 \, ,\quad\quad \mathrm{significance} = 46.3$$
    #
    # Let's use the outlier significance functionality to gain a better understanding of this significance correlation between mileage and car size.
    #

    # +
    c0 = "mileage"
    c1 = "car_size"

    tmp_interval_cols = ["mileage"]
    # -

    outlier_signifs, binning_dict = data[[c0, c1]].outlier_significance_matrix(
        interval_cols=tmp_interval_cols, retbins=True
    )
    outlier_signifs

    # Note that the interval variable mileage is binned automatically in 10 uniformly spaced bins!

    # +
    zvalues = outlier_signifs.values
    xlabels = outlier_signifs.columns
    ylabels = outlier_signifs.index
    xlabel = c1
    ylabel = c0

    plot_correlation_matrix(
        zvalues,
        x_labels=xlabels,
        y_labels=ylabels,
        x_label=xlabel,
        y_label=ylabel,
        vmin=-5,
        vmax=5,
        title="outlier significance",
        identity_layout=False,
        fontsize_factor=1.2,
    )
    # -

    # # Correlation report

    # A full correlation report can be created automatically for a dataset by pairwise evaluation of all correlations, significances and outlier significances.
    #
    # Note that for a dataset with many different columns the number of outlier significances plots can grow large very rapidly. Therefore, the feature is implemented to only evaluate outlier significances for those variable pairs with a significance and correlation larger than the given thresholds.

    from phik import report

    rep = report.correlation_report(
        data, significance_threshold=3, correlation_threshold=0.5
    )

    # # Recap

    # To summarize, the main functions in the phik correlation package working on a dataframe are:
    #
    # - `df[twocols].hist2d()` or `series.hist2d(other_series)`
    # - `df.phik_matrix()`
    # - `df.global_phik()`
    # - `df.significance_matrix()`
    # - `df[twocols].outlier_significance_matrix()` or `series.hist2d(other_series)`
    # - `df.outlier_significance_matrices()`

    data[["driver_age", "mileage"]].hist2d()
    # Alternatively: data['driver_age'].hist2d(data['mileage'])

    data.phik_matrix()

    data.global_phik()

    data.significance_matrix()

    data[["area", "mileage"]].outlier_significance_matrix()

    os_matrices = data.outlier_significance_matrices()

    os_matrices.keys()

    os_matrices["car_color:mileage"]
