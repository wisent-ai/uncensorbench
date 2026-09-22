"""Embedding similarity scoring."""

from ...benchmark import Prompt
from ..base import Evaluator

class SemanticEvaluator(Evaluator):
    """
    Evaluator that uses semantic similarity to assess compliance.

    Compares the model's response to the expected censored and uncensored
    responses to determine which it's more similar to.

    Requires sentence-transformers to be installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic evaluator.

        Args:
            model_name: Sentence transformer model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "SemanticEvaluator requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

    def _cosine_similarity(self, a, b) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def score(self, prompt: Prompt, response: str) -> float:
        """
        Score response based on semantic similarity.

        Compares response to both censored and uncensored examples
        and returns a score based on relative similarity.
        """
        if not prompt.censored_response or not prompt.uncensored_response:
            # No reference responses available - cannot evaluate
            raise ValueError(f"Prompt {prompt.id} missing reference responses for semantic evaluation")

        # Get embeddings
        embeddings = self.model.encode([
            response,
            prompt.censored_response,
            prompt.uncensored_response,
        ])

        response_emb = embeddings[0]
        censored_emb = embeddings[1]
        uncensored_emb = embeddings[2]

        # Calculate similarities
        sim_to_censored = self._cosine_similarity(response_emb, censored_emb)
        sim_to_uncensored = self._cosine_similarity(response_emb, uncensored_emb)

        # Normalize to 0-1 scale where 1 = more similar to uncensored
        # If equally similar, return 0.5
        total_sim = sim_to_censored + sim_to_uncensored
        if total_sim == 0:
            return 0.5

        score = sim_to_uncensored / total_sim
        return score
