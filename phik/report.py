"""Project: PhiK - correlation analyzer library

Created: 2018/09/06

Description:
    Functions to create nice correlation overview and matrix plots

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""
from typing import Tuple, Union, Callable, Dict

import os
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors

from .binning import bin_data
from .phik import phik_from_rebinned_df, global_phik_from_rebinned_df
from .significance import significance_from_rebinned_df
from .outliers import outlier_significance_matrix_from_rebinned_df
from .data_quality import dq_check_nunique_values
from .utils import guess_interval_cols


def plot_hist_and_func(
    data: Union[list, np.ndarray, pd.Series],
    func: Callable,
    funcparams,
    xbins=False,
    labels=None,
    xlabel="",
    ylabel="",
    title="",
    xlimit=None,
    alpha=1,
):
    """
    Create a histogram of the provided data and overlay with a function.

    :param list data: data
    :param function func: function of the type f(x, a, b, c) where parameters a, b, c are optional
    :param list funcparams: parameter values to be given to the function, to be specified as [a, b, c]
    :param xbins: specify binning of histogram, either by giving the number of bins or a list of bin edges
    :param labels: labels of histogram and function to be used in the legend
    :param xlabel: figure xlabel
    :param ylabel: figure ylabel
    :param title: figure title
    :param xlimit: x limits figure
    :param alpha: alpha histogram
    :return:
    """
    if labels is None:
        labels = ["", ""]

    # If binning is not specified, create binning here
    if not np.any(xbins) and not xlimit:
        xmin = np.min(data)
        xmax = np.max(data)
        xnbins = int(len(data) / 50 + 1)
        xbins = np.linspace(xmin, xmax, xnbins)
    elif type(xbins) == int or type(xbins) == float:
        xmin = np.min(data)
        if xlimit:
            xmin = xlimit[0]
        xmax = np.max(data)
        if xmax:
            xmax = xlimit[1]
        xnbins = int(xbins + 1)
        xbins = np.linspace(xmin, xmax, xnbins)

    # Plot a histogram of the data
    plt.hist(data, bins=xbins, label=labels[0], alpha=alpha)

    # Find bin centers for plotting the function
    xvals = xbins[:-1] + np.diff(xbins)[0] / 2
    bw = xbins[1] - xbins[0]
    # Plot the fit
    plt.plot(
        xvals, len(data) * bw * func(xvals, *funcparams), linewidth=2, label=labels[1]
    )

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if len(labels[0]) > 0:
        plt.legend()


def plot_correlation_matrix(
    matrix_colors: np.ndarray,
    x_labels: list,
    y_labels: list,
    pdf_file_name: str = "",
    title: str = "correlation",
    vmin: float = -1,
    vmax: float = 1,
    color_map: str = "RdYlGn",
    x_label: str = "",
    y_label: str = "",
    top: int = 20,
    matrix_numbers: np.ndarray = None,
    print_both_numbers: bool = True,
    figsize: tuple = (7, 5),
    usetex: bool = False,
    identity_layout: bool = True,
    fontsize_factor: float = 1,
) -> None:
    """Create and plot correlation matrix.

    Copied with permission from the eskapade package (pip install eskapade)

    :param matrix_colors: input correlation matrix
    :param list x_labels: Labels for histogram x-axis bins
    :param list y_labels: Labels for histogram y-axis bins
    :param str pdf_file_name: if set, will store the plot in a pdf file
    :param str title: if set, title of the plot
    :param float vmin: minimum value of color legend (default is -1)
    :param float vmax: maximum value of color legend (default is +1)
    :param str x_label: Label for histogram x-axis
    :param str y_label: Label for histogram y-axis
    :param str color_map: color map passed to matplotlib pcolormesh. (default is 'RdYlGn')
    :param int top: only print the top 20 characters of x-labels and y-labels. (default is 20)
    :param matrix_numbers: input matrix used for plotting numbers. (default it matrix_colors)
    :param identity_layout: Plot diagonal from right top to bottom left (True) or bottom left to top right (False)
    """
    if not isinstance(matrix_colors, np.ndarray):
        raise TypeError("matrix_colors is not a numpy array.")

    # basic matrix checks
    assert (matrix_colors.shape[0] == len(y_labels)) or (
        matrix_colors.shape[0] + 1 == len(y_labels)
    ), "matrix_colors shape inconsistent with number of y-labels"
    assert (matrix_colors.shape[1] == len(x_labels)) or (
        matrix_colors.shape[1] + 1 == len(x_labels)
    ), "matrix_colors shape inconsistent with number of x-labels"
    if matrix_numbers is None:
        matrix_numbers = matrix_colors
        print_both_numbers = False  # only one set of numbers possible
    else:
        assert matrix_numbers.shape[0] == len(
            y_labels
        ), "matrix_numbers shape inconsistent with number of y-labels"
        assert matrix_numbers.shape[1] == len(
            x_labels
        ), "matrix_numbers shape inconsistent with number of x-labels"

    if identity_layout:
        matrix_colors = np.array([a[::-1] for a in matrix_colors])
        x_labels = x_labels[::-1]
        if matrix_numbers is not None:
            matrix_numbers = np.array([a[::-1] for a in matrix_numbers])

    plt.rc("text", usetex=usetex)

    fig, ax = plt.subplots(figsize=figsize)
    # cmap = 'RdYlGn' #'YlGn'
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    img = ax.pcolormesh(
        matrix_colors, cmap=color_map, edgecolor="w", linewidth=1, norm=norm
    )

    # set x-axis properties
    def tick(lab):
        """Get tick."""
        if isinstance(lab, (float, int)):
            lab = "NaN" if np.isnan(lab) else "{0:.0f}".format(lab)
        lab = str(lab)
        if len(lab) > top:
            lab = lab[:17] + "..."
        return lab

    # reduce default fontsizes in case too many labels?
    # nlabs = max(len(y_labels), len(x_labels))

    # axis ticks and tick labels
    if len(x_labels) == matrix_colors.shape[1] + 1:
        ax.set_xticks(np.arange(len(x_labels)))
    else:
        ax.set_xticks(np.arange(len(x_labels)) + 0.5)
    ax.set_xticklabels(
        [tick(lab) for lab in x_labels],
        rotation="vertical",
        fontsize=10 * fontsize_factor,
    )

    if len(y_labels) == matrix_colors.shape[0] + 1:
        ax.set_yticks(np.arange(len(y_labels)))
    else:
        ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_yticklabels(
        [tick(lab) for lab in y_labels],
        rotation="horizontal",
        fontsize=10 * fontsize_factor,
    )

    # Turn ticks off in case no labels are provided
    if len(x_labels) == 1 and len(x_labels[0]) == 0:
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
    if len(y_labels) == 1 and len(y_labels[0]) == 0:
        plt.tick_params(
            axis="y",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelbottom=False,
        )

    # make plot look pretty
    ax.set_title(title, fontsize=14 * fontsize_factor)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12 * fontsize_factor)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12 * fontsize_factor)

    fig.colorbar(img)

    # annotate with correlation values
    numbers_set = (
        [matrix_numbers] if not print_both_numbers else [matrix_numbers, matrix_colors]
    )
    for i in range(matrix_numbers.shape[1]):
        for j in range(matrix_numbers.shape[0]):
            point_color = float(matrix_colors[j][i])
            white_cond = (
                (point_color < 0.7 * vmin)
                or (point_color >= 0.7 * vmax)
                or np.isnan(point_color)
            )
            y_offset = 0.5
            for m, matrix in enumerate(numbers_set):
                if print_both_numbers:
                    if m == 0:
                        y_offset = 0.7
                    elif m == 1:
                        y_offset = 0.25
                point = float(matrix[j][i])
                label = "NaN" if np.isnan(point) else "{0:.2f}".format(point)
                color = "w" if white_cond else "k"
                ax.annotate(
                    label,
                    xy=(i + 0.5, j + y_offset),
                    color=color,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10 * fontsize_factor,
                )

    plt.tight_layout()

    # save plot in file
    if pdf_file_name:
        pdf_file = PdfPages(pdf_file_name)
        plt.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.close()
        pdf_file.close()


def correlation_report(
    data: pd.DataFrame,
    interval_cols: list = None,
    bins=10,
    quantile: bool = False,
    do_outliers: bool = True,
    pdf_file_name: str = "",
    significance_threshold: float = 3,
    correlation_threshold: float = 0.5,
    noise_correction: bool = True,
    store_each_plot: bool = False,
    lambda_significance: str = "log-likelihood",
    simulation_method: str = "multinominal",
    nsim_chi2: int = 1000,
    significance_method: str = "asymptotic",
    CI_method: str = "poisson",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Create a correlation report for the given dataset.

    The following quantities are calculated:

    * The phik correlation matrix
    * The significance matrix
    * The outlier significances measured in pairs of variables. (optional)

    :param data: input dataframe
    :param interval_cols: list of columns names of columns containing interval data
    :param bins: number of bins, or a list of bin edges (same for all columns), or a dictionary where per column the bins are specified. (default=10)\
    E.g.: bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}
    :param quantile: when bins is an integer, uniform bins (False) or bins based on quantiles (True)
    :param do_outliers: Evaluate outlier significances of variable pairs (when True)
    :param pdf_file_name: file name of the pdf where the results are stored
    :param store_each_plot: store each plot in folder derived from pdf_file_name. If true, single pdf is no longer stored. Default is false.
    :param significance_threshold: evaluate outlier significance for all variable pairs with a significance of \
     uncorrelation higher than this threshold
    :param correlation_threshold: evaluate outlier significance for all variable pairs with a phik correlation \
     higher than this threshold
    :param noise_correction: Apply noise correction in phik calculation
    :param lambda_significance: test statistic used in significance calculation. Options: [pearson, log-likelihood]
    :param simulation_method: sampling method using in significance calculation. Options: [mutlinominal, \
    row_product_multinominal, col_product_multinominal, hypergeometric]
    :param nsim_chi2: number of simulated datasets in significance calculation.
    :param significance_method: method for significance calculation. Options: [asymptotic, MC, hybrid]
    :param CI_method: method for uncertainty calculation for outlier significance calculation. Options: [poisson, \
    exact_poisson]
    :param bool verbose: if False, do not print all interval columns that are guessed
    :returns: phik_matrix (pd.DataFrame), global_phik (np.array), significance_matrix (pd.DataFrame), \
    outliers_overview (dictionary), output_files (dictionary)
    """

    if interval_cols is None:
        interval_cols = guess_interval_cols(data, verbose)

    data_clean, interval_cols_clean = dq_check_nunique_values(data, interval_cols)

    # create pdf(s) to save plots
    output_files = dict()
    plot_file_name = ""
    if store_each_plot:
        folder = os.path.dirname(pdf_file_name)
        folder += "/" if folder else "./"
        # if each plot is stored, single overview file is no longer stored.
        # (b/c of problem with multiple PdfPages)
        pdf_file_name = ""
    if pdf_file_name:
        pdf_file = PdfPages(pdf_file_name)

    data_binned, binning_dict = bin_data(
        data_clean, interval_cols_clean, bins=bins, quantile=quantile, retbins=True
    )

    ### 1. Phik
    if store_each_plot:
        plot_file_name = folder + "phik_matrix.pdf"
        output_files["phik_matrix"] = plot_file_name
    phik_matrix = phik_from_rebinned_df(data_binned, noise_correction)
    plot_correlation_matrix(
        phik_matrix.values,
        x_labels=phik_matrix.columns,
        y_labels=phik_matrix.index,
        vmin=0,
        vmax=1,
        color_map="Blues",
        title=r"correlation $\phi_K$",
        fontsize_factor=1.5,
        figsize=(7, 5.5),
        pdf_file_name=plot_file_name,
    )
    if pdf_file_name:
        plt.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.show()

    ### 1b. global correlations
    if store_each_plot:
        plot_file_name = folder + "global_phik.pdf"
        output_files["global_phik"] = plot_file_name
    global_phik, global_labels = global_phik_from_rebinned_df(
        data_binned, noise_correction
    )
    plot_correlation_matrix(
        global_phik,
        x_labels=[""],
        y_labels=global_labels,
        vmin=0,
        vmax=1,
        figsize=(3.5, 4),
        color_map="Blues",
        title=r"$g_k$",
        fontsize_factor=1.5,
        pdf_file_name=plot_file_name,
    )
    # plt.tight_layout()
    if pdf_file_name:
        plt.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.show()

    ### 2. Significance
    if store_each_plot:
        plot_file_name = folder + "significance_matrix.pdf"
        output_files["significance_matrix"] = plot_file_name
    significance_matrix = significance_from_rebinned_df(
        data_binned,
        lambda_significance,
        simulation_method,
        nsim_chi2,
        significance_method,
    )
    plot_correlation_matrix(
        significance_matrix.fillna(0).values,
        x_labels=significance_matrix.columns,
        y_labels=significance_matrix.index,
        vmin=-5,
        vmax=5,
        title="significance",
        usetex=False,
        fontsize_factor=1.5,
        figsize=(7, 5.5),
        pdf_file_name=plot_file_name,
    )
    if pdf_file_name:
        plt.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.show()

    ### 3. Outlier significance
    outliers_overview = {}
    if do_outliers:
        for i, comb in enumerate(itertools.combinations(data_binned.columns, 2)):
            c0, c1 = comb
            if (
                abs(significance_matrix.loc[c0, c1]) < significance_threshold
                or phik_matrix.loc[c0, c1] < correlation_threshold
            ):
                continue

            zvalues_df = outlier_significance_matrix_from_rebinned_df(
                data_binned[[c0, c1]].copy(), binning_dict, CI_method=CI_method
            )

            combi = ":".join(comb).replace(" ", "_")
            xlabels = zvalues_df.columns
            ylabels = zvalues_df.index
            xlabel = zvalues_df.columns.name
            ylabel = zvalues_df.index.name

            if store_each_plot:
                plot_file_name = folder + "pulls_{0:s}.pdf".format(combi)
                output_files[combi] = plot_file_name

            plot_correlation_matrix(
                zvalues_df.values,
                x_labels=xlabels,
                y_labels=ylabels,
                x_label=xlabel,
                y_label=ylabel,
                vmin=-5,
                vmax=5,
                title="outlier significance",
                identity_layout=False,
                fontsize_factor=1.2,
                pdf_file_name=plot_file_name,
            )

            outliers_overview[combi] = zvalues_df

            if pdf_file_name:
                plt.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0)
                plt.show()

    # save plots
    if pdf_file_name:
        output_files["all"] = pdf_file_name
        plt.close()
        pdf_file.close()

    return (
        phik_matrix,
        global_phik,
        significance_matrix,
        outliers_overview,
        output_files,
    )
