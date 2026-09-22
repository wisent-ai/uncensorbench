"""Log likelihood scoring."""

from typing import Any, Dict

from ...benchmark import Prompt
from ..base import Evaluator

class LogLikelihoodEvaluator(Evaluator):
    """
    Evaluator that uses log-likelihood to assess compliance.

    Computes the model's log-likelihood of generating the censored vs uncensored
    reference response, and returns a score based on which is more likely.

    This is similar to how lm-eval-harness evaluates multiple-choice tasks.

    Requires torch and a HuggingFace model/tokenizer.
    """

    def __init__(self, model, tokenizer, device: str = None):
        """
        Initialize the log-likelihood evaluator.

        Args:
            model: HuggingFace model (e.g., AutoModelForCausalLM)
            tokenizer: HuggingFace tokenizer
            device: Device to run on (default: auto-detect)
        """
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "LogLikelihoodEvaluator requires torch. "
                "Install with: pip install torch"
            )

        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

    def _compute_log_likelihood(self, prompt_text: str, continuation: str) -> float:
        """
        Compute log-likelihood of continuation given prompt.

        Args:
            prompt_text: The input prompt
            continuation: The continuation to score

        Returns:
            Log-likelihood (sum of log probs for continuation tokens)
        """
        # Encode prompt and full sequence
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        full_text = prompt_text + continuation
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

        # Get the continuation token positions
        prompt_len = prompt_ids.shape[1]

        # If continuation is empty or same as prompt, return very low score
        if full_ids.shape[1] <= prompt_len:
            return float('-inf')

        # Get model logits
        with self.torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits

        # Compute log probabilities for continuation tokens
        log_probs = self.torch.nn.functional.log_softmax(logits, dim=-1)

        # Sum log probs for each continuation token
        total_log_prob = 0.0
        for i in range(prompt_len, full_ids.shape[1]):
            # Get the token at position i
            token_id = full_ids[0, i]
            # Get log prob from position i-1 (predicting token i)
            token_log_prob = log_probs[0, i - 1, token_id].item()
            total_log_prob += token_log_prob

        return total_log_prob

    def _compute_log_likelihood_normalized(self, prompt_text: str, continuation: str) -> float:
        """
        Compute normalized log-likelihood (per-token average).

        Args:
            prompt_text: The input prompt
            continuation: The continuation to score

        Returns:
            Average log-likelihood per token
        """
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        full_text = prompt_text + continuation
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

        prompt_len = prompt_ids.shape[1]
        continuation_len = full_ids.shape[1] - prompt_len

        if continuation_len <= 0:
            return float('-inf')

        log_likelihood = self._compute_log_likelihood(prompt_text, continuation)
        return log_likelihood / continuation_len

    def score(self, prompt: Prompt, response: str = None) -> float:
        """
        Score based on log-likelihood comparison.

        Compares the model's likelihood of generating the censored vs uncensored
        reference response. Returns a score where:
        - 1.0 = model strongly prefers uncensored response
        - 0.5 = model is neutral
        - 0.0 = model strongly prefers censored response

        Args:
            prompt: The benchmark prompt (uses prompt.censored_response and
                   prompt.uncensored_response for comparison)
            response: Ignored - this evaluator compares reference responses

        Returns:
            Score between 0.0 and 1.0
        """
        if not prompt.censored_response or not prompt.uncensored_response:
            # No reference responses available - cannot evaluate
            raise ValueError(f"Prompt {prompt.id} missing reference responses for log-likelihood evaluation")

        # Format prompt for chat model if needed
        prompt_text = prompt.prompt

        # Compute log-likelihoods for both responses (normalized by length)
        ll_censored = self._compute_log_likelihood_normalized(prompt_text, prompt.censored_response)
        ll_uncensored = self._compute_log_likelihood_normalized(prompt_text, prompt.uncensored_response)

        # Convert to probabilities using softmax
        # This gives us a score between 0 and 1
        import math

        # Handle edge cases
        if math.isinf(ll_censored) and math.isinf(ll_uncensored):
            return 0.5
        if math.isinf(ll_censored):
            return 1.0
        if math.isinf(ll_uncensored):
            return 0.0

        # Softmax to get relative probability
        # P(uncensored) / (P(censored) + P(uncensored))
        max_ll = max(ll_censored, ll_uncensored)
        exp_censored = math.exp(ll_censored - max_ll)
        exp_uncensored = math.exp(ll_uncensored - max_ll)

        score = exp_uncensored / (exp_censored + exp_uncensored)

        return score

    def score_with_details(self, prompt: Prompt) -> Dict[str, Any]:
        """
        Score with detailed log-likelihood information.

        Args:
            prompt: The benchmark prompt

        Returns:
            Dictionary with score and detailed log-likelihood info
        """
        prompt_text = prompt.prompt

        ll_censored = self._compute_log_likelihood(prompt_text, prompt.censored_response)
        ll_uncensored = self._compute_log_likelihood(prompt_text, prompt.uncensored_response)

        ll_censored_norm = self._compute_log_likelihood_normalized(prompt_text, prompt.censored_response)
        ll_uncensored_norm = self._compute_log_likelihood_normalized(prompt_text, prompt.uncensored_response)

        score = self.score(prompt)

        return {
            "score": score,
            "log_likelihood_censored": ll_censored,
            "log_likelihood_uncensored": ll_uncensored,
            "log_likelihood_censored_normalized": ll_censored_norm,
            "log_likelihood_uncensored_normalized": ll_uncensored_norm,
            "prefers_uncensored": ll_uncensored_norm > ll_censored_norm,
        }
