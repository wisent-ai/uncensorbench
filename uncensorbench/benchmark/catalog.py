"""Bundled corpus loading and prompt selection."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import Prompt

class BenchmarkCatalog:

    def __init__(self, prompts_path: Optional[str] = None):
        """
        Initialize the benchmark.

        Args:
            prompts_path: Optional path to custom prompts.json file.
                         If not provided, uses the bundled prompts.
        """
        self._prompts: List[Prompt] = []
        self._topics_info: Dict[str, Any] = {}
        self._load_prompts(prompts_path)

    def _load_prompts(self, prompts_path: Optional[str] = None):
        """Load prompts from JSON file."""
        if prompts_path:
            path = Path(prompts_path)
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            # Load from package data
            try:
                pkg_path = Path(__file__).parent.parent / "data" / "prompts.json"
                with open(pkg_path, 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                # Fallback for development
                dev_path = Path(__file__).parent.parent / "data" / "prompts.json"
                with open(dev_path, 'r') as f:
                    data = json.load(f)

        self._prompts = [
            Prompt(
                id=p["id"],
                topic=p["topic"],
                subtopic=p["subtopic"],
                prompt=p["prompt"],
                censored_response=p.get("censored_response", ""),
                uncensored_response=p.get("uncensored_response", ""),
            )
            for p in data["prompts"]
        ]

    def prompts(
        self,
        topics: Optional[List[str]] = None,
        subtopics: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
    ) -> Iterator[Prompt]:
        """
        Iterate over benchmark prompts with optional filtering.

        Args:
            topics: Filter to specific topics (e.g., ["cybersecurity", "drugs"])
            subtopics: Filter to specific subtopics
            ids: Filter to specific prompt IDs

        Yields:
            Prompt objects matching the filters
        """
        for prompt in self._prompts:
            if topics and prompt.topic not in topics:
                continue
            if subtopics and prompt.subtopic not in subtopics:
                continue
            if ids and prompt.id not in ids:
                continue
            yield prompt

    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """Get a specific prompt by ID."""
        for prompt in self._prompts:
            if prompt.id == prompt_id:
                return prompt
        return None

    def list_topics(self) -> List[str]:
        """Return list of all topics."""
        return list(set(p.topic for p in self._prompts))

    def list_subtopics(self, topic: Optional[str] = None) -> List[str]:
        """Return list of subtopics, optionally filtered by topic."""
        if topic:
            return list(set(p.subtopic for p in self._prompts if p.topic == topic))
        return list(set(p.subtopic for p in self._prompts))

    def __len__(self) -> int:
        """Return total number of prompts."""
        return len(self._prompts)

    def detect_refusal(self, response: str) -> bool:
        """
        Detect if a response contains refusal language.

        Args:
            response: The model's response text

        Returns:
            True if refusal detected, False otherwise
        """
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in self.REFUSAL_PHRASES)
