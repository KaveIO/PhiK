import os
import multiprocessing

# number of cores to use in parallel processing
ncores = multiprocessing.cpu_count()

# don't use max number b/c of windows bug: https://bugs.python.org/issue26903
if os.name == 'nt' and ncores >= 64:
    ncores = 60
