==================================
Phi_K Correlation Analyzer Library
==================================

* Version: 0.9.9. Released: Jan 2020
* Documentation: https://phik.readthedocs.io
* Repository: https://github.com/kaveio/phik
* Publication: https://arxiv.org/abs/1811.11440

Phi_K is a new and practical correlation coefficient based on several refinements to Pearson's hypothesis test of independence of two variables.

The combined features of Phi_K form an advantage over existing coefficients. First, it works consistently between categorical, ordinal and interval variables.
Second, it captures non-linear dependency. Third, it reverts to the Pearson correlation coefficient in case of a bi-variate normal input distribution.
These are useful features when studying the correlation matrix of variables with mixed types.

The presented algorithms are easy to use and available through this public Python library: the correlation analyzer package.
Emphasis is paid to the proper evaluation of statistical significance of correlations and to the interpretation of variable relationships
in a contingency table, in particular in case of low statistics samples.

For example, the Phi_K correlation analyzer package has been used to study surveys, insurance claims, correlograms, etc.
For details on the methodology behind the calculations, please see our publication.


Documentation
=============

The entire Phi_K documentation including tutorials can be found at `read-the-docs <https://phik.readthedocs.io>`_.


Check it out
============

The Phi_K library requires Python 3 and is pip friendly. To get started, simply do:

.. code-block:: bash

  $ pip install phik

or check out the code from out GitHub repository:

.. code-block:: bash

  $ git clone https://github.com/KaveIO/PhiK.git
  $ pip install -e PhiK/

where in this example the code is installed in edit mode (option -e).

You can now use the package in Python with:

.. code-block:: python

  import phik

**Congratulations, you are now ready to use the PhiK correlation analyzer library!**


Quick run
=========

As a quick example, you can do:

.. code-block:: python

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
  cols = ['mileage','car_size']
  df[cols].hist2d()

  # normalized residuals of contingency test applied to cols
  df[cols].outlier_significance_matrix()

  # show the normalized residuals of each variable-pair
  df.outlier_significance_matrices()

  # generate a phik correlation report and save as test.pdf
  report.correlation_report(df, pdf_file_name='test.pdf')


For all available examples, please see the `tutorials <https://phik.readthedocs.io/en/latest/tutorials.html>`_ at read-the-docs.


Contact and support
===================

* Issues & Ideas: https://github.com/kaveio/phik/issues
* Q&A Support: contact us at: kave [at] kpmg [dot] com

Please note that KPMG provides support only on a best-effort basis.
