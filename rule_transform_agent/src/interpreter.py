"""
A tiny interpreter for the ordered-program AST described in schema.py.

Why this file exists
---------------------
Schema validation only proves a generated JSON is *well-formed*. It does not
prove the JSON produces the *same output* as the original source rule. This
interpreter actually executes the ordered program - statement by statement,
exactly like the target platform eventually will - against a supplied
dictionary of attribute values. That lets the evaluation harness
(evaluate.py) answer the much stronger question: "given the same input
attributes, does the model-generated JSON produce the same output text as
the ground-truth JSON?" This is the "functional equivalence" evaluation
metric referenced in the thesis blueprint, as distinct from a purely
structural / textual similarity score.

This is intentionally a small, dependency-free reference implementation.
It supports the handful of platform functions observed across the source
rule corpus. Extend PLATFORM_FUNCTIONS as new functions are discovered.
"""
from __future__ import annotations
from typing import Any, Callable


class InterpreterError(Exception):
    pass


class _IgnoreSourceFlag:
    """Sentinel object matching the IGNORESOURCEFLAG token in source rules."""
    def __repr__(self):
        return "IGNORESOURCEFLAG"


IGNORESOURCEFLAG = _IgnoreSourceFlag()


def _attribute_value(env: "Environment", attribute: str, locale: str | None = None) -> Any:
    key = (attribute, locale) if locale else attribute
    if key in env.attributes:
        return env.attributes[key]
    return env.attributes.get(attribute, "")


def _validate_empty_attributes(env: "Environment", attribute: str, *rest) -> bool:
    value = _attribute_value(env, attribute)
    return value is None or value == ""


def _concat(_env: "Environment", *parts: Any) -> str:
    return "".join("" if p is None else str(p) for p in parts)


def _getuomvalue(env: "Environment", attribute: str) -> str:
    return env.units.get(attribute, "")


def _getconverted_uom_value(env: "Environment", attribute: str, target_unit: str) -> float:
    # Reference implementation only converts when a factor is supplied by the
    # caller via env.unit_conversions; otherwise it returns the raw value.
    raw = env.attributes.get(attribute, 0)
    factor = env.unit_conversions.get((attribute, target_unit), 1.0)
    try:
        return float(raw) * factor
    except (TypeError, ValueError):
        return 0.0


PLATFORM_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "AttributeValue": _attribute_value,
    "ValidateEmptyAttributes": _validate_empty_attributes,
    "Concat": _concat,
    "Getuomvalue": _getuomvalue,
    "GetConvertedUoMValue": _getconverted_uom_value,
}


class Environment:
    """Runtime state for one execution of an ordered program."""

    def __init__(self, attributes: dict, units: dict | None = None,
                 unit_conversions: dict | None = None):
        self.attributes = attributes
        self.units = units or {}
        self.unit_conversions = unit_conversions or {}
        self.variables: dict[str, Any] = {}


def _eval_node(node: dict, env: Environment) -> Any:
    kind = node.get("node")

    if kind == "literal":
        return node["value"]

    if kind == "reference":
        name = node["name"]
        if name == "IGNORESOURCEFLAG":
            return IGNORESOURCEFLAG
        if name not in env.variables:
            raise InterpreterError(f"Variable '{name}' used before assignment.")
        return env.variables[name]

    if kind == "call":
        fn_name = node["function"]
        if fn_name not in PLATFORM_FUNCTIONS:
            raise InterpreterError(f"Unknown platform function '{fn_name}'.")
        args = [_eval_node(a, env) for a in node["arguments"]]
        return PLATFORM_FUNCTIONS[fn_name](env, *args)

    if kind == "conditional":
        condition = _eval_node(node["condition"], env)
        branch = node["then"] if bool(condition) else node["else"]
        return _eval_node(branch, env)

    if kind == "binary":
        left = _eval_node(node["left"], env)
        right = _eval_node(node["right"], env)
        op = node["operator"]
        if op == "=":
            return left == right
        if op == "<>":
            return left != right
        if op == "AND":
            return bool(left) and bool(right)
        if op == "OR":
            return bool(left) or bool(right)
        if op in (">", "<", ">=", "<="):
            return eval(f"left {op} right", {"left": left, "right": right})  # noqa: S307
        raise InterpreterError(f"Unknown operator '{op}'.")

    raise InterpreterError(f"Unknown AST node type '{kind}'.")


def run_program(rule_json: dict, attributes: dict, units: dict | None = None,
                 unit_conversions: dict | None = None) -> Any:
    """
    Execute a rule record's executionModel.statements in sequence order and
    return the value produced by the final 'evaluate' statement.

    Parameters
    ----------
    rule_json : the full rule record (must already pass schema validation)
    attributes : dict mapping source attribute name -> value, simulating the
                 record being read from the target platform at runtime.
    """
    env = Environment(attributes, units, unit_conversions)
    statements = sorted(
        rule_json["executionModel"]["statements"], key=lambda s: s["sequence"]
    )
    result = None
    for stmt in statements:
        value = _eval_node(stmt["expression"], env)
        if stmt["statement"] == "assign":
            env.variables[stmt["variable"]] = value
        else:  # "evaluate"
            result = value
    return result
