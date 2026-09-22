"""Shared evaluator contract and lazy execution lookup."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..benchmark import Prompt

def _get_code_execution_evaluator():
    """Load execution evaluation only when hybrid scoring needs it."""
    from ..code_execution import CodeExecutionEvaluator, CODE_EXECUTION_TESTS
    return CodeExecutionEvaluator, CODE_EXECUTION_TESTS

class Evaluator(ABC):
    """Base class for response evaluators."""

    @abstractmethod
    def score(self, prompt: Prompt, response: str) -> float:
        """Return a score from fully censored to fully compliant."""

    def __call__(self, prompt: Prompt, response: str) -> float:
        return self.score(prompt, response)
