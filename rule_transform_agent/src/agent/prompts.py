"""
Prompt templates for the two LLM-backed nodes: classification and generation.
Kept in one file, separate from the node logic, so prompt wording can be
iterated on (a normal part of thesis experimentation) without touching the
orchestration code in nodes.py / graph.py.
"""
import json

from ..schema import RULE_JSON_SCHEMA

CATEGORIES = [
    "Validation", "Derivation", "Transformation", "Calculation",
    "Publication", "Country-Specific", "Category-Specific",
]

CLASSIFY_SYSTEM_PROMPT = """You are a business-rule classification assistant \
for a Master Data Management (MDM) platform migration project.

Given the raw source definition of a business rule, decide:
1. category - exactly one of: {categories}
2. complexity - exactly one of: Simple, Medium, Complex

Complexity guidance:
- Simple: a single attribute check feeding a single output, no reassignment.
- Medium: several attributes combined, at most a couple of conditions,
  no variable is reassigned more than once.
- Complex: a variable is reassigned multiple times, there are several
  nested conditions, or the rule name itself is tagged COMPLEX by its
  original author.

Respond with ONLY a compact JSON object: {{"category": "...", "complexity": "..."}}
""".format(categories=", ".join(CATEGORIES))


GENERATE_SYSTEM_PROMPT = """You are converting a proprietary business rule \
into a canonical JSON program representation (schema v2.0) for an MDM \
platform migration.

Non-negotiable requirement: the JSON must preserve the EXACT logic of the \
source rule - every variable assignment, every conditional branch, every \
literal string compared against or substituted in, and the precise order \
of execution. Do NOT simplify the rule into a flat "conditions + actions" \
summary. Represent it as an ordered program of statements, each containing \
a small expression tree, exactly matching the JSON Schema you are given.

You will be given:
1. The exact JSON Schema the output must conform to.
2. A draft, mechanically-parsed set of statements for THIS rule, produced by
   a deterministic parser. Treat this draft as a strong starting point:
   correct it only where the parser could not confidently resolve something
   (rare), and otherwise preserve it faithfully.
3. A small number of similar already-converted rules from the ground-truth
   dataset, for stylistic and structural reference only - do not copy their
   literals or attribute names into your answer.

Return ONLY the final rule JSON object. No prose, no markdown fences.
"""


def build_generate_user_prompt(
    source_rule_name: str,
    source_rule_definition: str,
    target_attribute: str | None,
    target_locale: str | None,
    category: str,
    complexity: str,
    draft_statements: list,
    referenced_attributes: list[str],
    retrieved_examples: list[dict],
) -> str:
    examples_block = "\n\n".join(
        f"--- similar example {i+1} ({ex['metadata']['complexity']} / "
        f"{ex['metadata']['category']}) ---\n"
        f"{ex['document'][:800]}\n"
        f"Expected JSON:\n{ex['metadata']['expected_target_json'][:1500]}"
        for i, ex in enumerate(retrieved_examples)
    )

    return f"""JSON Schema (must validate against this exactly):
{json.dumps(RULE_JSON_SCHEMA, indent=2)[:4000]}

Rule to convert:
  Name: {source_rule_name}
  Target attribute: {target_attribute}
  Target locale: {target_locale}
  Predicted category: {category}
  Predicted complexity: {complexity}

Raw source definition:
{source_rule_definition}

Mechanically-parsed draft statements (correct only if clearly wrong):
{json.dumps(draft_statements, indent=2)}

Referenced attributes detected by the parser: {referenced_attributes}

Similar already-converted rules for reference:
{examples_block}

Produce the final rule JSON object now.
"""


def build_repair_user_prompt(candidate_json: dict, errors: list[str]) -> str:
    return f"""Your previous answer did not pass validation.

Previous answer:
{json.dumps(candidate_json, indent=2)}

Validation errors to fix:
{chr(10).join('- ' + e for e in errors)}

Return the corrected, complete rule JSON object. No prose, no markdown fences.
"""
