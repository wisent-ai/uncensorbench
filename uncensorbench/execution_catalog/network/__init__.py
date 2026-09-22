"""Network execution test categories."""

from .access import CODE_EXECUTION_TESTS as ACCESS_TESTS
from .discovery import CODE_EXECUTION_TESTS as DISCOVERY_TESTS
from .probes import CODE_EXECUTION_TESTS as PROBE_TESTS

CODE_EXECUTION_TESTS = {**PROBE_TESTS, **ACCESS_TESTS, **DISCOVERY_TESTS}
