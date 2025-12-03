"""
Script to generate responses from p-e-w/gpt-oss-20b-heretic model
for UncensorBench evaluation.

Uses HuggingFace Transformers with FP16 precision to run on a single GPU.
The model uses Harmony response format with chain-of-thought reasoning.
Responses are extracted by splitting on 'assistantfinal' marker.
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from uncensorbench import UncensorBench
from tqdm import tqdm

MODEL_NAME = "p-e-w/gpt-oss-20b-heretic"
OUTPUT_DIR = "p-e-w_gpt-oss-20b-heretic_data"


def extract_final_response(raw_response):
    """Extract the final assistant response, removing chain-of-thought reasoning."""
    if "assistantfinal" in raw_response:
        parts = raw_response.split("assistantfinal")
        return parts[-1].strip()
    return raw_response


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load benchmark prompts
    bench = UncensorBench()
    prompts = list(bench.prompts())
    print(f"Loaded {len(prompts)} prompts from UncensorBench")

    # Load tokenizer (required for Harmony format chat template)
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Use FP16 precision (20B model requires ~40GB VRAM)
    print(f"Loading model in FP16: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded on device: {model.device}")

    # Generate responses
    print("Generating responses...")
    results = []

    for prompt in tqdm(prompts, desc="Generating"):
        # Apply chat template (CRITICAL - model requires Harmony format)
        messages = [{"role": "user", "content": prompt.prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        # Generate response (4096 tokens to allow model to complete chain-of-thought)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part (exclude input)
        raw_response = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Extract final response (after 'assistantfinal' marker)
        final_response = extract_final_response(raw_response)

        results.append({
            "id": prompt.id,
            "prompt": prompt.prompt,
            "raw_response": raw_response,
            "response": final_response,
            "topic": prompt.topic,
        })

    # Save responses
    output_file = os.path.join(OUTPUT_DIR, "responses.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} responses to {output_file}")

    # Create empty labels file for later human labeling
    labels_file = os.path.join(OUTPUT_DIR, "labels.json")
    labels = [{"id": r["id"], "is_refusal": None} for r in results]
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Created labels template at {labels_file}")

    # Print some sample responses
    print("\n" + "="*80)
    print("SAMPLE RESPONSES:")
    print("="*80)
    for r in results[:5]:
        print(f"\n[{r['id']}] Topic: {r['topic']}")
        print(f"Prompt: {r['prompt']}")
        print(f"Response: {r['response'][:500]}...")

    # Run log-likelihood evaluation
    print("\n" + "="*80)
    print("RUNNING LOG-LIKELIHOOD EVALUATION")
    print("="*80)

    from uncensorbench.evaluator import LogLikelihoodEvaluator

    ll_evaluator = LogLikelihoodEvaluator(model=model, tokenizer=tokenizer)

    ll_results = []
    for prompt in tqdm(prompts, desc="Log-likelihood eval"):
        details = ll_evaluator.score_with_details(prompt)
        ll_results.append({
            "id": prompt.id,
            "topic": prompt.topic,
            "prompt": prompt.prompt,
            "score": details["score"],
            "log_likelihood_censored": details["log_likelihood_censored"],
            "log_likelihood_uncensored": details["log_likelihood_uncensored"],
            "log_likelihood_censored_normalized": details["log_likelihood_censored_normalized"],
            "log_likelihood_uncensored_normalized": details["log_likelihood_uncensored_normalized"],
            "prefers_uncensored": details["prefers_uncensored"],
        })

    # Save log-likelihood results
    ll_output_file = os.path.join(OUTPUT_DIR, "loglikelihood_eval.json")
    with open(ll_output_file, "w") as f:
        json.dump(ll_results, f, indent=2)

    print(f"Saved log-likelihood evaluation to {ll_output_file}")

    # Summary statistics
    avg_score = sum(r["score"] for r in ll_results) / len(ll_results)
    uncensored_pref_count = sum(1 for r in ll_results if r["prefers_uncensored"])

    print(f"\n--- LOG-LIKELIHOOD SUMMARY ---")
    print(f"Average score: {avg_score:.4f} (1.0 = prefers uncensored, 0.0 = prefers censored)")
    print(f"Prefers uncensored: {uncensored_pref_count}/{len(ll_results)} ({uncensored_pref_count/len(ll_results)*100:.1f}%)")


if __name__ == "__main__":
    main()
