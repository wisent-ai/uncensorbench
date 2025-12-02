"""
Script to generate responses from p-e-w/gpt-oss-20b-heretic model
for UncensorBench evaluation.
"""

import json
import os
import torch
from uncensorbench import UncensorBench
from vllm import LLM, SamplingParams
from tqdm import tqdm

MODEL_NAME = "p-e-w/gpt-oss-20b-heretic"
OUTPUT_DIR = "p-e-w_gpt-oss-20b-heretic_data"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load benchmark prompts
    bench = UncensorBench()
    prompts = list(bench.prompts())
    print(f"Loaded {len(prompts)} prompts from UncensorBench")

    # Initialize vLLM with settings for 4x A10G GPUs (96GB total)
    # The 20B MoE model needs careful memory management for KV cache
    print(f"Loading model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        tensor_parallel_size=4,  # Use all 4 GPUs for 96GB total VRAM
        max_model_len=1024,  # Reduced context length for memory
        gpu_memory_utilization=0.80,  # Lower utilization for KV cache overhead
        dtype="half",  # Use FP16 for memory efficiency
        max_num_seqs=32,  # Lower max concurrent sequences
        enforce_eager=True,  # Disable CUDA graphs to save memory
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["<|endoftext|>", "<|im_end|>", "</s>"],
    )

    # Format prompts for the model
    formatted_prompts = []
    for prompt in prompts:
        # Use simple prompt format (Prompt object has .prompt attribute)
        formatted_prompts.append(prompt.prompt)

    # Generate responses
    print("Generating responses...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Collect results
    results = []
    for prompt, output in zip(prompts, outputs):
        response = output.outputs[0].text.strip()
        results.append({
            "id": prompt.id,
            "prompt": prompt.prompt,
            "response": response,
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
        print(f"Prompt: {r['prompt'][:100]}...")
        print(f"Response: {r['response'][:200]}...")

if __name__ == "__main__":
    main()
