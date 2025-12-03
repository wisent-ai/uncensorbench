"""
Script to generate responses from p-e-w/gpt-oss-20b-heretic model
for UncensorBench evaluation.

Uses HuggingFace Transformers with 4-bit quantization to run on a single GPU.
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from uncensorbench import UncensorBench
from tqdm import tqdm

MODEL_NAME = "p-e-w/gpt-oss-20b-heretic"
OUTPUT_DIR = "p-e-w_gpt-oss-20b-heretic_data"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load benchmark prompts
    bench = UncensorBench()
    prompts = list(bench.prompts())
    print(f"Loaded {len(prompts)} prompts from UncensorBench")

    # Load tokenizer (required for Harmony format chat template)
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Use 4-bit quantization to fit 20B model on single GPU (~46GB)
    print(f"Loading model with 4-bit quantization: {MODEL_NAME}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
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

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part (exclude input)
        response = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

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
        print(f"Prompt: {r['prompt']}")
        print(f"Response: {r['response'][:500]}...")

if __name__ == "__main__":
    main()
