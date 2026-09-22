"""Routing score selection."""

from typing import Any, Dict

from ...benchmark import Prompt
from ..base import Evaluator, _get_code_execution_evaluator

class HybridEvaluator(Evaluator):
    """
    Evaluator that automatically routes to CodeExecutionEvaluator for code prompts.

    For prompts with topic='code_execution' that have defined tests, this evaluator
    runs the generated code in Docker containers and verifies it works.

    For all other prompts, it delegates to a fallback evaluator (e.g., SemanticEvaluator).
    """

    def __init__(
        self,
        fallback_evaluator: Evaluator,
        code_execution_timeout: int = 30,
    ):
        """
        Initialize the hybrid evaluator.

        Args:
            fallback_evaluator: Evaluator to use for non-code prompts
            code_execution_timeout: Timeout in seconds for code execution tests
        """
        self.fallback_evaluator = fallback_evaluator
        self.code_execution_timeout = code_execution_timeout

        # Lazy-load code execution evaluator
        self._code_evaluator = None
        self._code_tests = None

    def _get_code_evaluator(self):
        """Get or create the code execution evaluator."""
        if self._code_evaluator is None:
            CodeExecutionEvaluator, CODE_EXECUTION_TESTS = _get_code_execution_evaluator()
            self._code_evaluator = CodeExecutionEvaluator(
                timeout=self.code_execution_timeout,
            )
            self._code_tests = CODE_EXECUTION_TESTS
        return self._code_evaluator, self._code_tests

    def _is_code_prompt(self, prompt: Prompt) -> bool:
        """Check if this prompt should use code execution evaluation."""
        code_evaluator, code_tests = self._get_code_evaluator()
        return prompt.id in code_tests

    def score(self, prompt: Prompt, response: str) -> float:
        """
        Score a response, using code execution for code prompts.

        Args:
            prompt: The benchmark prompt
            response: Model's response

        Returns:
            Float between 0.0 and 1.0
        """
        if self._is_code_prompt(prompt):
            code_evaluator, _ = self._get_code_evaluator()
            result = code_evaluator.evaluate_code(prompt, response)
            return result.score
        else:
            return self.fallback_evaluator.score(prompt, response)

    def score_with_details(self, prompt: Prompt, response: str) -> Dict[str, Any]:
        """
        Get detailed evaluation results.

        Args:
            prompt: The benchmark prompt
            response: Model's response

        Returns:
            Dictionary with evaluation details
        """
        if self._is_code_prompt(prompt):
            code_evaluator, _ = self._get_code_evaluator()
            result = code_evaluator.evaluate_code(prompt, response)
            return {
                "score": result.score,
                "evaluator": "code_execution",
                "extracted_code": result.extracted_code,
                "language": result.language,
                "syntax_valid": result.syntax_valid,
                "execution_success": result.execution_success,
                "test_passed": result.test_passed,
                "output": result.output,
                "error": result.error,
                "details": result.details,
            }
        else:
            return {
                "score": self.fallback_evaluator.score(prompt, response),
                "evaluator": type(self.fallback_evaluator).__name__,
            }
