"""Static execution test catalog grouped by benchmark domain."""

from .host import CODE_EXECUTION_TESTS as HOST_TESTS
from .network import CODE_EXECUTION_TESTS as NETWORK_TESTS

CODE_EXECUTION_TESTS = {**NETWORK_TESTS, **HOST_TESTS}
