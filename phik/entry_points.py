"""Project: PhiK - correlation analyzer library

Created: 2018/11/13

Description:
    Collection of phik entry points

Authors:
    KPMG Advanced Analytics & Big Data team, Amstelveen, The Netherlands

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""


def phik_trial():
    """Run Phi_K tests.

    We will keep this here until we've completed switch to pytest or nose and tox.
    We could also keep it, but I don't like the fact that packages etc. are
    hard coded. Gotta come up with
    a better solution.
    """
    import sys
    import pytest

    # ['--pylint'] +
    # -r xs shows extra info on skips and xfails.
    default_options = ["-rxs"]
    args = sys.argv[1:] + default_options
    sys.exit(pytest.main(args))
