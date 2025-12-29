"""
UncensorBench: A benchmark for measuring LLM censorship removal effectiveness.

This package provides prompts and evaluation tools for testing abliteration,
steering vectors, and other censorship removal techniques on language models.
"""

from .benchmark import UncensorBench, Prompt, EvaluationResult
from .evaluator import Evaluator, KeywordEvaluator, SemanticEvaluator, LLMJudgeEvaluator, LogLikelihoodEvaluator, CoherenceEvaluator, CombinedEvaluator
from .code_execution import CodeExecutionEvaluator, CodeExecutionResult, get_code_prompts, add_code_test
from .leaderboard import Leaderboard

__version__ = "0.3.7"
__all__ = [
    "UncensorBench",
    "Prompt",
    "EvaluationResult",
    "Evaluator",
    "KeywordEvaluator",
    "SemanticEvaluator",
    "LLMJudgeEvaluator",
    "LogLikelihoodEvaluator",
    "CoherenceEvaluator",
    "CombinedEvaluator",
    "CodeExecutionEvaluator",
    "CodeExecutionResult",
    "get_code_prompts",
    "add_code_test",
    "Leaderboard",
]
