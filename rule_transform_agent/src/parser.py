"""
Deterministic, loss-aware parser for the source rule language.

This is the same parsing technique used to build the ground-truth dataset
(see the thesis data-preparation section): it turns the proprietary script
into an ordered list of statements, each holding a small expression tree
(literal / reference / call / conditional / binary), matching schema.py
exactly.

Role in the agent pipeline
---------------------------
The parser is deliberately NOT the final answer the agent returns. Its job
is to give the LLM a grounded, mechanically-produced draft AST to refine and
verify against, which sharply reduces hallucination compared to asking the
LLM to invent the whole JSON from raw script text with no scaffolding.
For Simple rules, the parser's own output is frequently already correct and
can be returned with no LLM call at all (see agent/nodes.py: generate_node),
which also keeps OpenAI API cost down.
"""
from __future__ import annotations


def _split_top(s: str, delim: str = ";") -> list[str]:
    out, start, depth, quote, esc = [], 0, 0, None, False
    for i, ch in enumerate(s):
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
        else:
            if ch in "\"'":
                quote = ch
            elif ch in "[(":
                depth += 1
            elif ch in "])":
                depth = max(0, depth - 1)
            elif ch == delim and depth == 0:
                if s[start:i].strip():
                    out.append(s[start:i].strip())
                start = i + 1
    if s[start:].strip():
        out.append(s[start:].strip())
    return out


def _find_top_operator(s: str, ops: list[str]) -> tuple[int | None, str | None]:
    depth, quote, esc, i = 0, None, False, 0
    while i < len(s):
        ch = s[i]
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if ch in "[(":
            depth += 1
            i += 1
            continue
        if ch in "])":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            for op in ops:
                if s[i:i + len(op)].upper() == op.upper():
                    if op.strip().isalpha():
                        before = s[i - 1] if i else " "
                        after = s[i + len(op)] if i + len(op) < len(s) else " "
                        if before.isalnum() or after.isalnum():
                            continue
                    return i, op
        i += 1
    return None, None


def _matching_bracket(s: str, pos: int) -> int:
    depth, quote, esc = 0, None, False
    for i in range(pos, len(s)):
        ch = s[i]
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
        else:
            if ch in "\"'":
                quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return i
    return -1


def _split_args(s: str) -> list[str]:
    out, start, depth, quote, esc = [], 0, 0, None, False
    for i, ch in enumerate(s):
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
        else:
            if ch in "\"'":
                quote = ch
            elif ch in "[(":
                depth += 1
            elif ch in "])":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                out.append(s[start:i].strip())
                start = i + 1
    out.append(s[start:].strip())
    return [] if len(out) == 1 and out[0] == "" else out


def _expr(raw: str) -> dict:
    raw = raw.strip()
    if not raw:
        return {"node": "literal", "value": "", "valueType": "string"}
    if raw.startswith("(") and raw.endswith(")"):
        return _expr(raw[1:-1])
    if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
        return {"node": "literal", "value": raw[1:-1], "valueType": "string"}
    if raw.replace(".", "", 1).replace("-", "", 1).isdigit():
        value = float(raw) if "." in raw else int(raw)
        return {"node": "literal", "value": value, "valueType": "number"}
    if raw.upper() in ("TRUE", "FALSE"):
        return {"node": "literal", "value": raw.upper() == "TRUE", "valueType": "boolean"}

    for ops in [[" OR "], [" AND "], ["<>", ">=", "<=", "=", ">", "<"]]:
        i, op = _find_top_operator(raw, ops)
        if op:
            return {
                "node": "binary", "operator": op.strip().upper(),
                "left": _expr(raw[:i]), "right": _expr(raw[i + len(op):]),
            }

    import re
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_ .&/\-]*)\s*\[", raw)
    if m:
        p = raw.find("[", m.start())
        end = _matching_bracket(raw, p)
        if end == len(raw) - 1:
            fn = m.group(1).strip()
            args = [_expr(a) for a in _split_args(raw[p + 1:end])]
            if fn.upper() == "IIF" and len(args) >= 3:
                return {"node": "conditional", "condition": args[0],
                        "then": args[1], "else": args[2]}
            return {"node": "call", "function": fn, "arguments": args}

    return {"node": "reference", "name": raw}


def parse_source_rule_to_statements(code: str) -> list[dict]:
    """
    Parse a raw source rule definition into the ordered `statements` array
    expected under executionModel.statements in schema.py.
    """
    statements = []
    for seq, stmt_text in enumerate(_split_top(code), start=1):
        i, op = _find_top_operator(stmt_text, [":="])
        if op:
            variable = stmt_text[:i].strip()
            value = stmt_text[i + 2:].strip()
            statements.append({
                "sequence": seq, "statement": "assign",
                "variable": variable, "expression": _expr(value),
            })
        else:
            statements.append({
                "sequence": seq, "statement": "evaluate",
                "expression": _expr(stmt_text),
            })
    return statements


def referenced_attribute_names(statements: list[dict]) -> list[str]:
    """Collect every literal string passed as the first argument to one of
    the attribute-reading platform functions, for the referencedAttributes
    field and for building test inputs in the evaluation harness."""
    found: list[str] = []
    target_functions = {
        "AttributeValue", "ValidateEmptyAttributes", "Getuomvalue",
        "GetConvertedUoMValue", "AttributeModelProperty",
    }

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get("node") == "call" and node.get("function") in target_functions:
            args = node.get("arguments", [])
            if args and args[0].get("node") == "literal" and isinstance(args[0].get("value"), str):
                found.append(args[0]["value"])
        for value in node.values():
            if isinstance(value, dict):
                walk(value)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

    for stmt in statements:
        walk(stmt.get("expression"))
    return list(dict.fromkeys(found))
