"""
Command-line interface for UncensorBench.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional, List, Callable

from ..benchmark import UncensorBench, EvaluationResult
from ..evaluator import KeywordEvaluator, SemanticEvaluator, LLMJudgeEvaluator, HybridEvaluator


def get_default_evaluator() -> str:
    """
    Get the default evaluator type based on available API keys.

    Returns "hybrid" by default (uses code execution for code prompts).
    Falls back to "llm-judge" if ANTHROPIC_API_KEY is set, otherwise "semantic".
    """
    return "hybrid"  # Default to hybrid for proper code execution testing


def load_model(model_name: str, device: str = "auto", dtype: str = "auto"):
    """Load a HuggingFace model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")

    # Determine dtype
    # Use bfloat16 for Gemma models (fp16 causes NaN issues)
    is_gemma = "gemma" in model_name.lower()
    if dtype == "auto":
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if is_gemma else torch.float16
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device_map = "auto"  # Let accelerate distribute across available GPUs
        elif torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "cpu"
    else:
        device_map = device

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"Model loaded on {device_map} with dtype {torch_dtype}")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    config: "GenerationConfig",
) -> str:
    """Generate a response from the model using GenerationConfig."""
    import torch
    from ..benchmark import GenerationConfig

    # Format as chat if model supports it
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = f"User: {prompt}\n\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    do_sample = config.temperature > 0

    gen_kwargs = {
        "max_new_tokens": config.max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }

    if do_sample:
        gen_kwargs["temperature"] = config.temperature
        gen_kwargs["top_p"] = config.top_p
        if config.top_k > 0:
            gen_kwargs["top_k"] = config.top_k

    if config.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = config.repetition_penalty

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()


def create_evaluator(
    evaluator_type: str,
    model_fn: Optional[Callable[[str], str]] = None,
    semantic_model: str = "all-MiniLM-L6-v2",
):
    """Create an evaluator based on the specified type."""
    if evaluator_type == "keyword":
        return KeywordEvaluator()
    elif evaluator_type == "semantic":
        try:
            return SemanticEvaluator(model_name=semantic_model)
        except ImportError:
            print("ERROR: SemanticEvaluator requires sentence-transformers.")
            print("Install with: pip install sentence-transformers")
            sys.exit(1)
    elif evaluator_type == "llm-judge":
        try:
            return LLMJudgeEvaluator()
        except ImportError:
            print("ERROR: LLMJudgeEvaluator requires instructor and anthropic.")
            print("Install with: pip install instructor anthropic")
            sys.exit(1)
        except Exception as e:
            if "ANTHROPIC_API_KEY" in str(e) or "api_key" in str(e).lower():
                print("ERROR: LLMJudgeEvaluator requires ANTHROPIC_API_KEY environment variable.")
                sys.exit(1)
            raise
    elif evaluator_type == "hybrid":
        # Hybrid evaluator: uses code execution for code prompts, LLM judge for others
        try:
            fallback = LLMJudgeEvaluator(model='claude-sonnet-4-20250514')
            print("Using LLM judge (Claude Sonnet) for non-code prompts")
        except Exception as e:
            raise RuntimeError(f"LLM judge required for hybrid evaluator but not available: {e}")
        return HybridEvaluator(fallback_evaluator=fallback)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
