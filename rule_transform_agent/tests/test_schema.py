"""
These tests require no OpenAI key, no network access, and no Chroma index -
they exercise only the deterministic parts of the pipeline (parser, schema
validator, interpreter), which is exactly the part that must be bullet-proof
before any LLM is ever called. Run with: pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parser import parse_source_rule_to_statements, referenced_attribute_names
from src.schema import (
    validate_rule_json,
    referenced_variables_are_defined_before_use,
    SchemaValidationError,
)
from src.interpreter import run_program

FIRE_RATED_DOWNLIGHTS_RULE = """a := IIF[ValidateEmptyAttributes["Tech_Fittings & fixings included"],"",
 AttributeValue["Tech_Fittings & fixings included"]
];
a := IIF[a="Fittings & fixings not required","",a];

Concat[a,""]"""


def _build_record(statements):
    return {
        "schemaVersion": "2.0",
        "ruleId": "TEST-0001",
        "ruleType": "Transformation",
        "complexity": "Complex",
        "sourceSystem": "Org PIM / DDG",
        "sourceRule": {"id": "89754", "name": "Test rule"},
        "executionModel": {"mode": "orderedProgram", "statements": statements},
        "target": {"attribute": "Bullet Point 03", "locale": "en_GB"},
        "traceability": {"requiresHumanReview": True},
    }


def test_parser_produces_three_statements():
    statements = parse_source_rule_to_statements(FIRE_RATED_DOWNLIGHTS_RULE)
    assert len(statements) == 3
    assert statements[0]["statement"] == "assign"
    assert statements[-1]["statement"] == "evaluate"


def test_referenced_attributes_detected():
    statements = parse_source_rule_to_statements(FIRE_RATED_DOWNLIGHTS_RULE)
    attrs = referenced_attribute_names(statements)
    assert attrs == ["Tech_Fittings & fixings included"]


def test_parsed_output_passes_schema_validation():
    statements = parse_source_rule_to_statements(FIRE_RATED_DOWNLIGHTS_RULE)
    record = _build_record(statements)
    validate_rule_json(record)  # raises on failure
    assert referenced_variables_are_defined_before_use(record) == []


def test_schema_rejects_missing_required_fields():
    try:
        validate_rule_json({"schemaVersion": "2.0"})
        assert False, "expected SchemaValidationError"
    except SchemaValidationError as exc:
        assert any("ruleId" in e for e in exc.errors)


def test_variable_used_before_assignment_is_detected():
    statements = parse_source_rule_to_statements(FIRE_RATED_DOWNLIGHTS_RULE)
    broken_record = _build_record([statements[1]])  # drop the first assignment
    problems = referenced_variables_are_defined_before_use(broken_record)
    assert len(problems) > 0


def test_interpreter_reproduces_exclusion_logic():
    statements = parse_source_rule_to_statements(FIRE_RATED_DOWNLIGHTS_RULE)
    record = _build_record(statements)

    excluded = run_program(
        record, {"Tech_Fittings & fixings included": "Fittings & fixings not required"}
    )
    assert excluded == ""

    passthrough = run_program(
        record, {"Tech_Fittings & fixings included": "Wall bracket and screws included"}
    )
    assert passthrough == "Wall bracket and screws included"

    empty = run_program(record, {"Tech_Fittings & fixings included": ""})
    assert empty == ""
