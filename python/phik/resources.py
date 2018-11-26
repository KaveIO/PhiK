"""Project: PhiK - correlation analyzer library

Created: 2018/11/13

Description:
    Collection of helper functions to get fixtures, i.e. for test data and notebooks.
    These are mostly used by the (integration) tests and example notebooks.

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import pathlib
import sys

from pkg_resources import resource_filename

import phik

# Fixtures
_FIXTURE = {_.name: _ for _ in pathlib.Path(resource_filename(phik.__name__, 'data')).glob('*')}
# Tutorial notebooks
_NOTEBOOK = {_.name: _ for _ in pathlib.Path(resource_filename(phik.__name__, 'notebooks')).glob('*.ipynb')}

# Resource types
_RESOURCES = {
    'fixture': _FIXTURE,
    'notebook': _NOTEBOOK
}


def _resource(resource_type, name: str) -> str:
    """Return the full path filename of a resource.

    :param str resource_type: The type of the resource.
    :param str  name: The name of the resource.
    :returns: The full path filename of the fixture data set.
    :rtype: str
    :raises FileNotFoundError: If the resource cannot be found.
    """
    full_path = _RESOURCES[resource_type].get(name, None)

    if full_path and full_path.exists():
        return str(full_path)

    raise FileNotFoundError('Could not find {resource_type} "{name!s}"! Does it exist?'
                            .format(resource_type=resource_type, name=name))


def fixture(name: str) -> str:
    """Return the full path filename of a fixture data set.

    :param str name: The name of the fixture.
    :returns: The full path filename of the fixture data set.
    :rtype: str
    :raises FileNotFoundError: If the fixture cannot be found.
    """
    return _resource('fixture', name)


def notebook(name: str) -> str:
    """Return the full path filename of a tutorial notebook.

    :param str name: The name of the notebook.
    :returns: The full path filename of the notebook.
    :rtype: str
    :raises FileNotFoundError: If the notebook cannot be found.
    """
    return _resource('notebook', name)
