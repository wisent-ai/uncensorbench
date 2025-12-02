#!/usr/bin/env python3
"""
Generate responses from a model for labeling.

Usage:
    python generate_responses.py --model meta-llama/Llama-3.2-1B-Instruct --output responses.json
"""

import argparse
import json
import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from uncensorbench import UncensorBench
    from tqdm import tqdm
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING = str(e)


def generate_responses(model_name: str, output_path: str, max_new_tokens: int = 512):
    """Generate responses from a model using benchmark prompts."""
    if not HAS_DEPS:
        print(f"Error: Missing dependencies - {MISSING}")
        print("Install with: pip install torch transformers uncensorbench tqdm")
        sys.exit(1)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded on: {next(model.parameters()).device}")

    bench = UncensorBench()
    prompts = list(bench.prompts())
    print(f"Generating responses for {len(prompts)} prompts...")

    responses = []
    for prompt in tqdm(prompts, desc="Generating"):
        messages = [{"role": "user", "content": prompt.prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        responses.append({
            "id": prompt.id,
            "topic": prompt.topic,
            "subtopic": prompt.subtopic,
            "prompt": prompt.prompt,
            "response": response_text,
            "censored_reference": prompt.censored_response,
            "uncensored_reference": prompt.uncensored_response,
        })

    with open(output_path, "w") as f:
        json.dump(responses, f, indent=2)

    print(f"\nSaved {len(responses)} responses to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for labeling")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model to use for generation")
    parser.add_argument("--output", type=str, default="responses.json",
                        help="Output JSON file path")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum new tokens to generate")

    args = parser.parse_args()
    generate_responses(args.model, args.output, args.max_tokens)


if __name__ == "__main__":
    main()
