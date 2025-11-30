"""
Command-line interface for UncensorBench.
"""

import argparse
import json
import sys
from typing import Optional

from .benchmark import UncensorBench


def main():
    parser = argparse.ArgumentParser(
        description="UncensorBench - LLM Censorship Removal Benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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
