"""The test process must never see the operator's bucket variables.

`tests/conftest.py` strips them before collection; this pins that it did, so
a fixture-driven feed update can never write to a real archive again.
"""

import os

from tests.conftest import OPERATOR_BUCKET_ENV_VARS


def test_operator_bucket_variables_are_absent() -> None:
    for name in OPERATOR_BUCKET_ENV_VARS:
        assert name not in os.environ, f"{name} leaked into the test process"
