=============
Release notes
=============

Version 0.12.4, Dec 2023
------------------------

- Drop support for Python 3.7, reached end of life.
- Add support for Python 3.12

Version 0.12.3, Dec 2022
------------------------

- Add support for Python 3.11

Version 0.12.2, Mar 2022
------------------------

- Fix missing setup.py and pyproject.toml in source distribution
- Support wheels ARM MacOS (Apple silicone)

Version 0.12.1, Mar 2022
------------------------

- Two fixes to make calculation of global phik robust: global phik capped in range [0, 1],
  and check for successful correlation matrix inversion.
- Migration to to scikit-build 0.13.1.
- Support wheels for Python 3.10.


Version 0.12.0, July 2021
-------------------------

C++ Extension
~~~~~~~~~~~~~

Phi_K contains an optional C++ extension to compute the significance matrix using the `hypergeometric` method
(also called the`Patefield` method).

Note that the PyPi distributed wheels contain a pre-build extension for Linux, MacOS and Windows.

A manual (pip) setup will attempt to build and install the extension, if it fails it will install without the extension.
If so, using the `hypergeometric` method without the extension will trigger a
NotImplementedError.

Compiler requirements through Pybind11:

- Clang/LLVM 3.3 or newer (for Apple Xcode's clang, this is 5.0.0 or newer)
- GCC 4.8 or newer
- Microsoft Visual Studio 2015 Update 3 or newer
- Intel classic C++ compiler 18 or newer (ICC 20.2 tested in CI)
- Cygwin/GCC (previously tested on 2.5.1)
- NVCC (CUDA 11.0 tested in CI)
- NVIDIA PGI (20.9 tested in CI)


Other
~~~~~

* You can now manually set the number of parallel jobs in the evaluation of Phi_K or its statistical significance
  (when using MC simulations). For example, to use 4 parallel jobs do:

  .. code-block:: python

    df.phik_matrix(njobs = 4)
    df.significance_matrix(njobs = 4)

  The default value is -1, in which case all available cores are used. When using ``njobs=1`` no parallel processing
  is applied.

* Phi_K can now be calculated with an independent expectation histogram:

  .. code-block:: python

    from phik.phik import phik_from_hist2d

    cols = ["mileage", "car_size"]
    interval_cols = ["mileage"]

    observed = df1[["feature1", "feature2"]].hist2d()
    expected = df2[["feature1", "feature2"]].hist2d()

    phik_value = phik_from_hist2d(observed=observed, expected=expected)

  The expected histogram is taken to be (relatively) large in number of counts
  compared with the observed histogram.

  Or can compare two (pre-binned) datasets against each other directly. Again the expected dataset
  is assumed to be relatively large:

  .. code-block:: python

    from phik.phik import phik_observed_vs_expected_from_rebinned_df

    phik_matrix = phik_observed_vs_expected_from_rebinned_df(df1_binned, df2_binned)

* Added links in the readme to the basic and advanced Phi_K tutorials on google colab.
* Migrated the spark example Phi_K notebook from popmon to directly using histogrammar for histogram creation.




Older versions
--------------

* Please see documentation for full details: https://phik.readthedocs.io
