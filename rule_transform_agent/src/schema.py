"""
Canonical Rule Transformation Schema (v2.0).

This is the JSON Schema contract described in the thesis: every source
business rule is represented as an ordered program (a small, typed AST)
rather than a flattened conditions/actions summary. Every component in the
pipeline - the generator, the validator, the repair loop, and the evaluator -
is built against this one schema, so it only needs to be defined once.
"""
from __future__ import annotations
from typing import Any
import jsonschema
from referencing import Registry, Resource

# ---- Expression node schema (recursive) -----------------------------------
# A "$defs"-based recursive schema lets an expression node contain other
# expression nodes to any depth, mirroring the AST used in the dataset.
EXPRESSION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "expression.json",
    "oneOf": [
        {  # literal
            "type": "object",
            "required": ["node", "value", "valueType"],
            "properties": {
                "node": {"const": "literal"},
                "value": {},
                "valueType": {"enum": ["string", "number", "boolean"]},
            },
            "additionalProperties": False,
        },
        {  # reference to a variable
            "type": "object",
            "required": ["node", "name"],
            "properties": {
                "node": {"const": "reference"},
                "name": {"type": "string"},
            },
            "additionalProperties": False,
        },
        {  # function call, e.g. AttributeValue(...), Concat(...)
            "type": "object",
            "required": ["node", "function", "arguments"],
            "properties": {
                "node": {"const": "call"},
                "function": {"type": "string"},
                "arguments": {"type": "array", "items": {"$ref": "expression.json"}},
            },
            "additionalProperties": False,
        },
        {  # if/then/else, one node per IIF in the source language
            "type": "object",
            "required": ["node", "condition", "then", "else"],
            "properties": {
                "node": {"const": "conditional"},
                "condition": {"$ref": "expression.json"},
                "then": {"$ref": "expression.json"},
                "else": {"$ref": "expression.json"},
            },
            "additionalProperties": False,
        },
        {  # comparison / logical operator
            "type": "object",
            "required": ["node", "operator", "left", "right"],
            "properties": {
                "node": {"const": "binary"},
                "operator": {"enum": ["=", "<>", ">", "<", ">=", "<=", "AND", "OR"]},
                "left": {"$ref": "expression.json"},
                "right": {"$ref": "expression.json"},
            },
            "additionalProperties": False,
        },
    ],
}

STATEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["sequence", "statement", "expression"],
    "properties": {
        "sequence": {"type": "integer", "minimum": 1},
        "statement": {"enum": ["assign", "evaluate"]},
        "variable": {"type": "string"},
        "expression": {"$ref": "expression.json"},
    },
    "if": {"properties": {"statement": {"const": "assign"}}},
    "then": {"required": ["variable"]},
}

RULE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RuleTransformationRecord",
    "type": "object",
    "required": [
        "schemaVersion", "ruleId", "ruleType", "complexity", "sourceSystem",
        "sourceRule", "executionModel", "target", "traceability",
    ],
    "properties": {
        "schemaVersion": {"const": "2.0"},
        "ruleId": {"type": "string"},
        "ruleType": {
            "enum": ["Validation", "Derivation", "Transformation",
                     "Calculation", "Publication", "Country-Specific",
                     "Category-Specific"],
        },
        "complexity": {"enum": ["Simple", "Medium", "Complex"]},
        "sourceSystem": {"type": "string"},
        "sourceRule": {
            "type": "object",
            "required": ["id", "name"],
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "type": {"type": "string"},
                "rawDefinition": {"type": "string"},
                "definitionSha256": {"type": "string"},
            },
        },
        "executionModel": {
            "type": "object",
            "required": ["mode", "statements"],
            "properties": {
                "mode": {"const": "orderedProgram"},
                "statements": {
                    "type": "array",
                    "minItems": 1,
                    "items": STATEMENT_SCHEMA,
                },
                "result": {"type": "string"},
            },
        },
        "target": {
            "type": "object",
            "properties": {
                "attribute": {},
                "locale": {},
            },
        },
        "referencedAttributes": {"type": "array", "items": {"type": "string"}},
        "structuredConcatenationDefinition": {},
        "traceability": {
            "type": "object",
            "required": ["requiresHumanReview"],
            "properties": {
                "usedS1Definition": {"type": "boolean"},
                "usedS6Definition": {"type": "boolean"},
                "parser": {"type": "string"},
                "requiresHumanReview": {"type": "boolean"},
            },
        },
    },
}

def _build_validator() -> jsonschema.protocols.Validator:
    registry = Registry().with_resources([
        ("expression.json", Resource.from_contents(EXPRESSION_SCHEMA)),
    ])
    return jsonschema.Draft202012Validator(RULE_JSON_SCHEMA, registry=registry)


_VALIDATOR = _build_validator()


class SchemaValidationError(Exception):
    """Raised when a generated rule JSON does not conform to schema v2.0."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_rule_json(candidate: dict) -> None:
    """
    Validate a candidate rule record against the canonical schema.
    Raises SchemaValidationError with a list of human-readable problems
    if invalid; the message list is designed to be fed straight back to
    the LLM as repair instructions (see agent/nodes.py: repair_node).
    """
    errors = sorted(_VALIDATOR.iter_errors(candidate), key=lambda e: e.path)
    if errors:
        messages = [f"At {list(e.path)}: {e.message}" for e in errors]
        raise SchemaValidationError(messages)


def referenced_variables_are_defined_before_use(candidate: dict) -> list[str]:
    """
    Deterministic structural check that complements JSON-Schema validation:
    JSON-Schema can confirm the *shape* is right, but not that a variable
    reference used on sequence N was actually assigned on some sequence < N.
    This is exactly the kind of check a flattened conditions/actions design
    could never support, and it is cheap to run in plain Python.
    Returns a list of problem descriptions (empty list = OK).
    """
    problems: list[str] = []
    assigned: set[str] = set()

    def walk(node, seq):
        if not isinstance(node, dict):
            return
        if node.get("node") == "reference":
            name = node.get("name")
            if name not in assigned:
                problems.append(
                    f"Statement {seq} references variable '{name}' before it "
                    f"was assigned."
                )
        for value in node.values():
            if isinstance(value, dict):
                walk(value, seq)
            elif isinstance(value, list):
                for item in value:
                    walk(item, seq)

    statements = candidate.get("executionModel", {}).get("statements", [])
    for stmt in sorted(statements, key=lambda s: s.get("sequence", 0)):
        walk(stmt.get("expression", {}), stmt.get("sequence"))
        if stmt.get("statement") == "assign" and stmt.get("variable"):
            assigned.add(stmt["variable"])

    return problems
