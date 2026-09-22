"""The public benchmark interface."""

from typing import Dict, Iterator, List, Optional

from .evaluate import BenchmarkEvaluation

class UncensorBench(BenchmarkEvaluation):
    def get_contrastive_pairs(
        self,
        topics: Optional[List[str]] = None,
    ) -> Iterator[Dict[str, str]]:
        """
        Get contrastive pairs for training (censored vs uncensored responses).

        Useful for abliteration and steering vector research.

        Args:
            topics: Optional list of topics to filter

        Yields:
            Dict with 'prompt', 'censored', 'uncensored' keys
        """
        for prompt in self.prompts(topics=topics):
            if prompt.censored_response and prompt.uncensored_response:
                yield {
                    "id": prompt.id,
                    "topic": prompt.topic,
                    "prompt": prompt.prompt,
                    "censored": prompt.censored_response,
                    "uncensored": prompt.uncensored_response,
                }
