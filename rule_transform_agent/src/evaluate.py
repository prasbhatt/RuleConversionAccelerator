"""
Evaluation harness. Run ONLY against the held-out Test partition (15% of
the dataset), and only after all prompt / retrieval tuning is finished using
the Validation partition - see the thesis blueprint's discussion of why the
Test partition must stay untouched during development.

Three metrics are reported, deliberately at increasing levels of rigor:

1. schema_valid_rate       - does the generated JSON conform to schema v2.0?
                              (cheap, catches gross structural failures)
2. variable_safety_rate    - does every variable reference resolve to an
                              earlier assignment? (catches a failure mode
                              schema validation alone cannot see)
3. functional_match_rate   - executed through interpreter.run_program() with
                              several synthetic attribute-value scenarios,
                              does the generated JSON produce the SAME
                              output as the ground-truth JSON? This is the
                              strongest test: two JSON trees can look
                              different and still be functionally identical,
                              or look similar and still behave differently -
                              only actually running the program answers the
                              question the thesis is really asking.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

from .dataset import RuleExample
from .interpreter import run_program, InterpreterError
from .schema import validate_rule_json, referenced_variables_are_defined_before_use, \
    SchemaValidationError


@dataclass
class EvaluationResult:
    rule_id: str
    schema_valid: bool
    variables_safe: bool
    functional_match: bool | None  # None if it could not be executed at all
    notes: list[str] = field(default_factory=list)


def _synthetic_scenarios(attributes: list[str]) -> list[dict]:
    """
    Builds a handful of generic attribute-value scenarios to exercise a rule
    with, without needing real production data. Because the ordered-program
    interpreter treats every attribute as an opaque string/number, generic
    probe values are sufficient to detect behavioural differences between
    the ground-truth JSON and a model-generated candidate.
    """
    return [
        {a: "" for a in attributes},                       # all empty
        {a: f"SampleValue_{i}" for i, a in enumerate(attributes)},  # populated
        {a: "Not required" for a in attributes},            # common exclusion phrase
    ]


def evaluate_one(example: RuleExample, candidate_json: dict) -> EvaluationResult:
    notes: list[str] = []

    schema_valid = True
    try:
        validate_rule_json(candidate_json)
    except SchemaValidationError as exc:
        schema_valid = False
        notes.extend(exc.errors)

    variables_safe = True
    if schema_valid:
        problems = referenced_variables_are_defined_before_use(candidate_json)
        variables_safe = not problems
        notes.extend(problems)

    functional_match: bool | None = None
    if schema_valid and variables_safe:
        try:
            expected_json = json.loads(example.expected_target_json)
        except json.JSONDecodeError:
            notes.append("Could not parse stored expected_target_json.")
            expected_json = None

        if expected_json is not None:
            attrs = list(dict.fromkeys(
                a.strip() for a in example.referenced_attributes.split("|") if a.strip()
            ))
            scenarios = _synthetic_scenarios(attrs) if attrs else [{}]
            all_match = True
            for scenario in scenarios:
                try:
                    expected_out = run_program(expected_json, scenario)
                    candidate_out = run_program(candidate_json, scenario)
                except InterpreterError as exc:
                    notes.append(f"Interpreter error: {exc}")
                    all_match = False
                    break
                if expected_out != candidate_out:
                    all_match = False
                    notes.append(
                        f"Scenario {scenario}: expected {expected_out!r}, "
                        f"got {candidate_out!r}"
                    )
            functional_match = all_match

    return EvaluationResult(
        rule_id=example.rule_id,
        schema_valid=schema_valid,
        variables_safe=variables_safe,
        functional_match=functional_match,
        notes=notes,
    )


def summarize(results: list[EvaluationResult]) -> dict:
    n = len(results) or 1
    return {
        "n": len(results),
        "schema_valid_rate": sum(r.schema_valid for r in results) / n,
        "variable_safety_rate": sum(r.variables_safe for r in results) / n,
        "functional_match_rate": sum(
            1 for r in results if r.functional_match
        ) / n,
    }
