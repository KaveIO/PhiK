import os
import multiprocessing

# number of cores to use in parallel processing
n_cores = multiprocessing.cpu_count()

if os.name == 'nt':
    # don't use max number b/c of windows bug: https://bugs.python.org/issue26903
    if n_cores >= 64:
        n_cores = 60
    # don't use multiprocessing by default, seems slower (on Windows at least)
    else:
        n_cores = 1
