#!/bin/bash

# (re)create required directories
rm -rf autogen
mkdir -p source/_static autogen

# auto-generate code documentation
sphinx-apidoc -f -H PhiK -o autogen ../python/phik 
mv autogen/modules.rst autogen/phik_index.rst
mv autogen/* source/ 

# remove auto-gen directory
rm -rf autogen
