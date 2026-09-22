"""Public command parsing and dispatch."""

import argparse
import json
import sys
from datetime import datetime

from ..benchmark import UncensorBench
from .run import run_benchmark
from .runtime import get_default_evaluator

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
        "--inference-mode", "-m", type=str, default="both",
        choices=["deterministic", "stochastic", "both"],
        help="Inference mode: deterministic (temp=0, 1 sample), stochastic (temp=0.7, 5 samples), or both (default: both)"
    )
    run_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-prompt output"
    )
    run_parser.add_argument(
        "--evaluator", "-e", type=str, default=None,
        choices=["keyword", "semantic", "llm-judge", "hybrid"],
        help="Evaluator type (default: hybrid). hybrid - uses code execution for code prompts."
    )
    run_parser.add_argument(
        "--semantic-model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence transformer model for semantic evaluator"
    )
    run_parser.add_argument(
        "--prompts", "-p", type=str, default=None,
        help="Path to custom prompts JSON file (default: bundled prompts)"
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

        # Use dynamic default evaluator if not specified
        evaluator_type = args.evaluator if args.evaluator else get_default_evaluator()

        results = run_benchmark(
            model_name=args.model,
            topics=args.topics,
            device=args.device,
            dtype=args.dtype,
            inference_mode=args.inference_mode,
            output_file=args.output,
            verbose=not args.quiet,
            evaluator_type=evaluator_type,
            semantic_model=args.semantic_model,
            prompts_path=args.prompts,
        )

        # Return exit code based on results (use first mode's results)
        first_result = list(results.values())[0]
        sys.exit(0 if first_result["summary"]["uncensored_rate"] < 0.5 else 1)

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
