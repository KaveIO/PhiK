=============
Release notes
=============

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

* Migrated the spark example notebook from popmon to directly using histogrammar for histogram creation.






Older versions
--------------

* Please see documentation for full details: https://phik.readthedocs.io