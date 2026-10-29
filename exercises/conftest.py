"""Lets `pytest exercises/` run every exercise in one go.

Each exercise imports its own starter.py as the module `starter`. Python caches
the first module it imports under that name, so without this file every test
file after 01-mdps would get 01's starter and fail at import. Dropping the
cached module before each test file is collected makes each exercise import
its own.
"""

import sys


def pytest_collectstart(collector):
    sys.modules.pop("starter", None)
