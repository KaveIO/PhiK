"""Project: Phi_K - correlation coefficient package

Created: 2018/11/13

Description:
    setup script to install Phi_K correlation package.

Authors:
    KPMG Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import sys
from warnings import warn
from setuptools import find_packages
from setuptools import setup
from skbuild import setup as sk_setup
import pybind11

NAME = 'phik'

MAJOR = 0
REVISION = 12
PATCH = 3
DEV = False

# note: also update README.rst, CHANGES.rst

VERSION = '{major}.{revision}.{patch}'.format(major=MAJOR, revision=REVISION, patch=PATCH)
FULL_VERSION = VERSION
if DEV:
    FULL_VERSION += '.dev'

TEST_REQUIREMENTS = [
    'pytest>=4.0.2',
    'pytest-pylint>=0.13.0',
    'nbconvert>=5.3.1',
    'jupyter_client>=5.2.3',
]

REQUIREMENTS = [
    'numpy>=1.18.0',
    'scipy>=1.5.2',
    'pandas>=0.25.1',
    'matplotlib>=2.2.3',
    'joblib>=0.14.1',
]

EXTRA_REQUIREMENTS = {
    'test': TEST_REQUIREMENTS,
}

if DEV:
    REQUIREMENTS += TEST_REQUIREMENTS

EXCLUDE_PACKAGES = []

# read the contents of readme file
with open("README.rst", encoding="utf-8") as f:
    long_description = f.read()


def write_version_py(filename: str = 'phik/version.py') -> None:
    """Write package version to version.py.

    This will ensure that the version in version.py is in sync with us.

    :param filename: The version.py to write too.
    :type filename: str
    :return:
    :rtype: None
    """
    # Do not modify the indentation of version_str!
    version_str = """\"\"\"THIS FILE IS AUTO-GENERATED BY PHIK SETUP.PY.\"\"\"

name = '{name!s}'
version = '{version!s}'
full_version = '{full_version!s}'
release = {is_release!s}
"""

    with open(filename, 'w') as version_file:
        version_file.write(
            version_str.format(name=NAME.lower(), version=VERSION, full_version=FULL_VERSION, is_release=not DEV)
        )


setup_args = {
    'name': NAME,
    'version': FULL_VERSION,
    'url': 'http://phik.rtfd.io',
    'license': 'Apache-2',
    'author': 'KPMG N.V. The Netherlands',
    'author_email': 'kave@kpmg.com',
    'description': "Phi_K correlation analyzer library",
    'long_description': long_description,
    'long_description_content_type': "text/x-rst",
    'python_requires': '>=3.7',
    'packages': find_packages(exclude=EXCLUDE_PACKAGES),
    # Setuptools requires that package data are located inside the package.
    # This is a feature and not a bug, see
    # http://setuptools.readthedocs.io/en/latest/setuptools.html#non-package-data-files
    'package_data': {
        NAME.lower(): ['data/*', 'notebooks/phik_tutorial*.ipynb', ]
    },
    'include_package_data': True,
    'install_requires': REQUIREMENTS,
    'extras_require': EXTRA_REQUIREMENTS,
    'tests_require': TEST_REQUIREMENTS,
    'zip_safe': False,
    'classifiers': [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    # The following 'creates' executable scripts for *nix and Windows.
    # As an added bonus the Windows scripts will auto-magically
    # get a .exe extension.
    #
    # phik_trial: test application to let loose on tests. This is just a wrapper around pytest.
    'entry_points': {
        'console_scripts': [
            'phik_trial = phik.entry_points:phik_trial'
        ]
    }
}

sk_build_kwargs = {
    'cmake_args': [
        f"-Dpybind11_DIR:STRING={pybind11.get_cmake_dir()}",
        "-DPYTHON_EXECUTABLE={}".format(sys.executable),
        f"-DPHIK_VERSION_INFO={VERSION}",
    ]
}

if __name__ == '__main__':
    write_version_py()
    try:
        # try building with C++ extension:
        sk_setup(**setup_args, **sk_build_kwargs)
    except Exception as ex:
        warn(
            '\n---------------------------------------------\n'
            'WARNING\n\n'
            'The Phi_K C++ extension could not be compiled\n\n'
            f'{ex.__class__.__name__} {ex.__str__()}\n\n'
            '\n---------------------------------------------\n'
        )

        # # Retry to install the module without extension :
        # If this new 'setup' call doesn't fail, the module
        # will be successfully installed, without the C++ extension :
        setup(**setup_args)
