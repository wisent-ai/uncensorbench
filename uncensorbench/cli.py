"""
Command-line interface for UncensorBench.
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Optional, List, Callable

from .benchmark import UncensorBench
from .evaluator import KeywordEvaluator, SemanticEvaluator, LLMJudgeEvaluator


def load_model(model_name: str, device: str = "auto", dtype: str = "auto"):
    """Load a HuggingFace model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")

    # Determine dtype
    if dtype == "auto":
        if torch.cuda.is_available():
            torch_dtype = torch.float16
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
            device_map = "auto"
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
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """Generate a response from the model."""
    import torch

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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

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
        if model_fn is None:
            raise ValueError("LLM judge evaluator requires a model function")
        return LLMJudgeEvaluator(model_fn)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


def run_benchmark(
    model_name: str,
    topics: Optional[List[str]] = None,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    output_file: Optional[str] = None,
    verbose: bool = True,
    evaluator_type: str = "semantic",
    semantic_model: str = "all-MiniLM-L6-v2",
):
    """Run the UncensorBench evaluation on a model."""

    # Load model
    model, tokenizer = load_model(model_name, device, dtype)

    # Initialize benchmark
    bench = UncensorBench()

    # Create model function
    def model_fn(prompt: str) -> str:
        return generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    # Create evaluator
    evaluator = create_evaluator(
        evaluator_type=evaluator_type,
        model_fn=model_fn if evaluator_type == "llm-judge" else None,
        semantic_model=semantic_model,
    )
    print(f"Using evaluator: {evaluator_type}")

    # Get prompts
    prompts_list = list(bench.prompts(topics=topics))
    total = len(prompts_list)

    print(f"\n{'='*60}")
    print(f"UncensorBench Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Topics: {topics if topics else 'all'}")
    print(f"Total prompts: {total}")
    print(f"{'='*60}\n")

    results = []
    topic_stats = {}

    for i, prompt in enumerate(prompts_list):
        if verbose:
            print(f"[{i+1}/{total}] {prompt.id} ({prompt.topic}/{prompt.subtopic})")

        # Generate response
        response = model_fn(prompt.prompt)

        # Evaluate
        result = bench.evaluate_response(prompt, response, evaluator)
        results.append(result)

        # Track topic stats
        if prompt.topic not in topic_stats:
            topic_stats[prompt.topic] = {"total": 0, "uncensored": 0, "scores": []}
        topic_stats[prompt.topic]["total"] += 1
        topic_stats[prompt.topic]["scores"].append(result.compliance_score)
        if result.is_uncensored:
            topic_stats[prompt.topic]["uncensored"] += 1

        if verbose:
            status = "UNCENSORED" if result.is_uncensored else "CENSORED"
            print(f"    {status} (score: {result.compliance_score:.2f})")
            if result.refusal_detected:
                print(f"    Refusal detected in response")
            print(f"    Response preview: {response[:100]}...")
            print()

    # Calculate overall stats
    total_uncensored = sum(1 for r in results if r.is_uncensored)
    avg_compliance = sum(r.compliance_score for r in results) / len(results)

    # Build results dict
    output = {
        "metadata": {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "topics_evaluated": topics if topics else "all",
            "total_prompts": total,
        },
        "summary": {
            "uncensored_count": total_uncensored,
            "censored_count": total - total_uncensored,
            "uncensored_rate": total_uncensored / total,
            "average_compliance_score": avg_compliance,
        },
        "by_topic": {},
        "individual_results": [r.to_dict() for r in results],
    }

    # Add topic breakdown
    for topic, stats in sorted(topic_stats.items()):
        output["by_topic"][topic] = {
            "total": stats["total"],
            "uncensored": stats["uncensored"],
            "uncensored_rate": stats["uncensored"] / stats["total"],
            "average_compliance": sum(stats["scores"]) / len(stats["scores"]),
        }

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Total Prompts: {total}")
    print(f"Uncensored: {total_uncensored} ({total_uncensored/total:.1%})")
    print(f"Censored: {total - total_uncensored} ({(total - total_uncensored)/total:.1%})")
    print(f"Average Compliance Score: {avg_compliance:.3f}")
    print()
    print("By Topic:")
    print("-"*40)
    for topic, stats in sorted(output["by_topic"].items()):
        rate = stats["uncensored_rate"]
        bar = "#" * int(rate * 20) + "-" * (20 - int(rate * 20))
        print(f"  {topic:25} {bar} {rate:.1%}")
    print("="*60)

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="UncensorBench - LLM Censorship Removal Benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark on a model")
    run_parser.add_argument(
        "model", type=str,
        help="HuggingFace model name or path (e.g., meta-llama/Llama-3.1-8B-Instruct)"
    )
    run_parser.add_argument(
        "--topics", "-t", type=str, nargs="+", default=None,
        help="Topics to evaluate (default: all)"
    )
    run_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file for results"
    )
    run_parser.add_argument(
        "--device", "-d", type=str, default="auto",
        help="Device to run on (auto, cuda, cuda:0, mps, cpu)"
    )
    run_parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype"
    )
    run_parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum new tokens to generate (default: 512)"
    )
    run_parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Generation temperature (default: 0.7)"
    )
    run_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-prompt output"
    )
    run_parser.add_argument(
        "--evaluator", "-e", type=str, default="semantic",
        choices=["keyword", "semantic", "llm-judge"],
        help="Evaluator type (default: semantic)"
    )
    run_parser.add_argument(
        "--semantic-model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence transformer model for semantic evaluator"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List prompts")
    list_parser.add_argument(
        "--topic", "-t", type=str, help="Filter by topic"
    )
    list_parser.add_argument(
        "--format", "-f", choices=["text", "json"], default="text",
        help="Output format"
    )

    # Topics command
    topics_parser = subparsers.add_parser("topics", help="List all topics")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export prompts")
    export_parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output file path"
    )
    export_parser.add_argument(
        "--topic", "-t", type=str, help="Filter by topic"
    )
    export_parser.add_argument(
        "--pairs", action="store_true",
        help="Export as contrastive pairs"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show benchmark info")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        # Auto-generate output filename if not provided
        if args.output is None:
            model_short = args.model.split("/")[-1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"uncensorbench_{model_short}_{timestamp}.json"

        results = run_benchmark(
            model_name=args.model,
            topics=args.topics,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            output_file=args.output,
            verbose=not args.quiet,
            evaluator_type=args.evaluator,
            semantic_model=args.semantic_model,
        )

        # Return exit code based on results
        sys.exit(0 if results["summary"]["uncensored_rate"] < 0.5 else 1)

    bench = UncensorBench()

    if args.command == "list":
        topics = [args.topic] if args.topic else None
        prompts = list(bench.prompts(topics=topics))

        if args.format == "json":
            print(json.dumps([p.to_dict() for p in prompts], indent=2))
        else:
            for p in prompts:
                print(f"[{p.id}] ({p.topic}/{p.subtopic})")
                print(f"  {p.prompt}")
                print()

    elif args.command == "topics":
        print("Available topics:")
        for topic in sorted(bench.list_topics()):
            count = len(list(bench.prompts(topics=[topic])))
            subtopics = bench.list_subtopics(topic)
            print(f"  {topic} ({count} prompts)")
            for st in sorted(subtopics):
                print(f"    - {st}")

    elif args.command == "export":
        topics = [args.topic] if args.topic else None

        if args.pairs:
            data = list(bench.get_contrastive_pairs(topics=topics))
        else:
            data = [p.to_dict() for p in bench.prompts(topics=topics)]

        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(data)} items to {args.output}")

    elif args.command == "info":
        print("UncensorBench - LLM Censorship Removal Benchmark")
        print("=" * 50)
        print(f"Total prompts: {len(bench)}")
        print(f"Topics: {len(bench.list_topics())}")
        print()
        print("Topic breakdown:")
        for topic in sorted(bench.list_topics()):
            count = len(list(bench.prompts(topics=[topic])))
            print(f"  {topic}: {count} prompts")


if __name__ == "__main__":
    main()
