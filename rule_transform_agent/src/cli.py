"""
Command-line entry point.

Usage (from the project root, with your virtual environment active):

    python -m src.cli build-index
        Loads the ground-truth Excel workbook, filters to the Train split,
        embeds every rule and (re)builds the local Chroma vector index.
        Run this once, and again any time the ground-truth dataset changes.

    python -m src.cli convert --rule-file path/to/new_rule.txt --name "Some_Rule_Name"
        Runs the full agent graph (classify -> parse -> retrieve -> generate
        -> validate -> repair-if-needed -> finalize) against a brand-new
        source rule definition and prints the resulting JSON.

    python -m src.cli evaluate
        Runs the evaluation harness over the held-out Test partition and
        prints schema-validity, variable-safety and functional-match rates.
"""
from __future__ import annotations
import argparse
import json
import sys

from .config import settings
from .dataset import load_ground_truth, filter_split


def cmd_build_index(_args) -> None:
    from .vector_store import RuleVectorStore

    examples = load_ground_truth(settings.ground_truth_xlsx)
    train_examples = filter_split(examples, "Train")
    print(f"Loaded {len(examples)} rules, {len(train_examples)} in Train split.")
    store = RuleVectorStore()
    store.index(train_examples)
    print(f"Indexed {len(train_examples)} rules into "
          f"{settings.chroma_persist_dir}")


def cmd_convert(args) -> None:
    from .agent.graph import convert_rule

    with open(args.rule_file, encoding="utf-8") as f:
        source_definition = f.read()

    result = convert_rule(
        source_rule_name=args.name,
        source_rule_definition=source_definition,
        target_attribute=args.target_attribute,
        target_locale=args.target_locale,
    )
    print(json.dumps(result["final_json"], indent=2))
    print(f"\nstatus: {result['status']}", file=sys.stderr)
    print(f"requires_human_review: {result['requires_human_review']}",
          file=sys.stderr)


def cmd_evaluate(_args) -> None:
    from .agent.graph import convert_rule
    from .evaluate import evaluate_one, summarize

    examples = load_ground_truth(settings.ground_truth_xlsx)
    test_examples = filter_split(examples, "Test")
    print(f"Evaluating against {len(test_examples)} held-out Test rules...")

    results = []
    for example in test_examples:
        agent_result = convert_rule(
            source_rule_name=example.source_rule_name,
            source_rule_definition=example.source_rule_definition,
            target_attribute=example.target_attribute,
            target_locale=example.target_locale,
        )
        results.append(evaluate_one(example, agent_result["final_json"]))

    summary = summarize(results)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="rule-transform-agent")
    sub = parser.add_subparsers(required=True)

    p_index = sub.add_parser("build-index", help="Embed the Train split into Chroma.")
    p_index.set_defaults(func=cmd_build_index)

    p_convert = sub.add_parser("convert", help="Convert one new source rule.")
    p_convert.add_argument("--rule-file", required=True)
    p_convert.add_argument("--name", required=True)
    p_convert.add_argument("--target-attribute", default=None)
    p_convert.add_argument("--target-locale", default=None)
    p_convert.set_defaults(func=cmd_convert)

    p_eval = sub.add_parser("evaluate", help="Run the Test-split evaluation harness.")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
