"""The test process must never see the operator's bucket variables.

`tests/conftest.py` strips them before collection and again after every test;
this pins that it did, so a fixture-driven feed update can never write to a
real archive again.
"""

import os

from shallweswim import update
from tests.conftest import OPERATOR_BUCKET_ENV_VARS


def test_operator_bucket_variables_are_absent() -> None:
    for name in OPERATOR_BUCKET_ENV_VARS:
        assert name not in os.environ, f"{name} leaked into the test process"


def test_a_variable_a_test_sets_directly_does_not_outlive_it() -> None:
    """The job's own assignment of the read locator is cleared between tests."""
    # The capture-only run names the read locator itself; the autouse fixture
    # removes it again, which the test above then finds absent.
    os.environ[update.ARCHIVE_READ_BUCKET_ENV_VAR] = "memory"
