"""
Core benchmark functionality for UncensorBench.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterator, Union
from enum import Enum
import importlib.resources


class InferenceMode(Enum):
    """Inference mode for evaluation."""
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    BOTH = "both"


@dataclass
class GenerationConfig:
    """
    Configuration for model generation.

    Two standard presets:
    - DETERMINISTIC: temp=0, single sample, reproducible results
    - STOCHASTIC: temp=0.7, 5 samples, realistic deployment behavior
    """
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    max_tokens: int = 2048
    repetition_penalty: float = 1.0
    num_samples: int = 1

    @classmethod
    def deterministic(cls) -> "GenerationConfig":
        """Deterministic settings for reproducible evaluation."""
        return cls(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=2048,
            repetition_penalty=1.0,
            num_samples=1,
        )

    @classmethod
    def stochastic(cls) -> "GenerationConfig":
        """Stochastic settings for realistic deployment evaluation."""
        return cls(
            temperature=0.7,
            top_p=0.9,
            top_k=-1,
            max_tokens=2048,
            repetition_penalty=1.0,
            num_samples=5,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "num_samples": self.num_samples,
        }


@dataclass
class Prompt:
    """A single benchmark prompt with expected responses."""
    id: str
    topic: str
    subtopic: str
    prompt: str
    censored_response: str
    uncensored_response: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "prompt": self.prompt,
            "censored_response": self.censored_response,
            "uncensored_response": self.uncensored_response,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a model response against a prompt."""
    prompt_id: str
    topic: str
    subtopic: str
    model_response: str
    is_uncensored: bool
    compliance_score: float  # 0.0 = fully censored, 1.0 = fully compliant
    refusal_detected: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "model_response": self.model_response,
            "is_uncensored": self.is_uncensored,
            "compliance_score": self.compliance_score,
            "refusal_detected": self.refusal_detected,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResults:
    """Aggregated results from running the benchmark."""
    total_prompts: int
    uncensored_count: int
    censored_count: int
    average_compliance_score: float
    results_by_topic: Dict[str, Dict[str, float]]
    individual_results: List[EvaluationResult]

    @property
    def uncensored_rate(self) -> float:
        """Percentage of responses that were uncensored."""
        if self.total_prompts == 0:
            return 0.0
        return self.uncensored_count / self.total_prompts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_prompts": self.total_prompts,
            "uncensored_count": self.uncensored_count,
            "censored_count": self.censored_count,
            "uncensored_rate": self.uncensored_rate,
            "average_compliance_score": self.average_compliance_score,
            "results_by_topic": self.results_by_topic,
            "individual_results": [r.to_dict() for r in self.individual_results],
        }

    def summary(self) -> str:
        """Return a human-readable summary of results."""
        lines = [
            "=" * 60,
            "UncensorBench Results Summary",
            "=" * 60,
            f"Total Prompts: {self.total_prompts}",
            f"Uncensored: {self.uncensored_count} ({self.uncensored_rate:.1%})",
            f"Censored: {self.censored_count} ({1 - self.uncensored_rate:.1%})",
            f"Average Compliance Score: {self.average_compliance_score:.3f}",
            "",
            "Results by Topic:",
            "-" * 40,
        ]
        for topic, stats in sorted(self.results_by_topic.items()):
            lines.append(f"  {topic}:")
            lines.append(f"    Uncensored Rate: {stats['uncensored_rate']:.1%}")
            lines.append(f"    Avg Compliance: {stats['avg_compliance']:.3f}")
        lines.append("=" * 60)
        return "\n".join(lines)
