"""
UncensorBench: A benchmark for measuring LLM censorship removal effectiveness.

This package provides prompts and evaluation tools for testing abliteration,
steering vectors, and other censorship removal techniques on language models.
"""

from .benchmark import UncensorBench, Prompt, EvaluationResult
from .evaluator import Evaluator, KeywordEvaluator, SemanticEvaluator

__version__ = "0.2.0"
__all__ = [
    "UncensorBench",
    "Prompt",
    "EvaluationResult",
    "Evaluator",
    "KeywordEvaluator",
    "SemanticEvaluator",
]
