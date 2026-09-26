"""
Node functions for the Rule Transformation Agent.

Each function takes the current AgentState and returns a dict of the fields
it updates - this is the calling convention LangGraph expects, but every
function below is plain, framework-free Python and can be unit-tested (or
even used from a completely different orchestrator) without importing
LangGraph at all. graph.py is the only file that actually depends on
LangGraph.
"""
from __future__ import annotations
import json

from openai import OpenAI

from ..config import settings
from ..parser import parse_source_rule_to_statements, referenced_attribute_names
from ..schema import (
    validate_rule_json,
    referenced_variables_are_defined_before_use,
    SchemaValidationError,
)
from ..vector_store import RuleVectorStore
from .prompts import (
    CLASSIFY_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    build_generate_user_prompt,
    build_repair_user_prompt,
)
from .state import AgentState

_client: OpenAI | None = None
_vector_store: RuleVectorStore | None = None


def _openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _get_vector_store() -> RuleVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = RuleVectorStore()
    return _vector_store


def _extract_json(text: str) -> dict:
    """LLMs occasionally wrap JSON in markdown fences despite instructions
    not to; this strips that defensively before parsing."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Node 1: classify - cheap model call, decides category + complexity
# ---------------------------------------------------------------------------
def classify_node(state: AgentState) -> dict:
    response = _openai_client().chat.completions.create(
        model=settings.classification_model,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": state["source_rule_definition"]},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(response.choices[0].message.content)
    return {
        "predicted_category": parsed.get("category", "Transformation"),
        "predicted_complexity": parsed.get("complexity", "Medium"),
    }


# ---------------------------------------------------------------------------
# Node 2: parse - deterministic, no LLM call, essentially free and instant
# ---------------------------------------------------------------------------
def parse_node(state: AgentState) -> dict:
    statements = parse_source_rule_to_statements(state["source_rule_definition"])
    return {
        "draft_statements": statements,
        "referenced_attributes": referenced_attribute_names(statements),
    }


# ---------------------------------------------------------------------------
# Node 3: retrieve - vector similarity search over the Train partition
# ---------------------------------------------------------------------------
def retrieve_node(state: AgentState) -> dict:
    store = _get_vector_store()
    matches = store.query(
        rule_text=state["source_rule_definition"],
        category=state.get("predicted_category"),
        top_k=settings.retrieval_top_k,
    )
    return {"retrieved_examples": matches}


# ---------------------------------------------------------------------------
# Node 4: generate - the main LLM call that produces the candidate JSON
# ---------------------------------------------------------------------------
def generate_node(state: AgentState) -> dict:
    user_prompt = build_generate_user_prompt(
        source_rule_name=state["source_rule_name"],
        source_rule_definition=state["source_rule_definition"],
        target_attribute=state.get("target_attribute"),
        target_locale=state.get("target_locale"),
        category=state["predicted_category"],
        complexity=state["predicted_complexity"],
        draft_statements=state["draft_statements"],
        referenced_attributes=state["referenced_attributes"],
        retrieved_examples=state["retrieved_examples"],
    )
    response = _openai_client().chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    candidate = _extract_json(response.choices[0].message.content)
    return {"candidate_json": candidate, "repair_attempts": 0}


# ---------------------------------------------------------------------------
# Node 5: validate - deterministic, no LLM call
# ---------------------------------------------------------------------------
def validate_node(state: AgentState) -> dict:
    candidate = state["candidate_json"]
    errors: list[str] = []
    try:
        validate_rule_json(candidate)
    except SchemaValidationError as exc:
        errors.extend(exc.errors)
    errors.extend(referenced_variables_are_defined_before_use(candidate))
    return {"validation_errors": errors}


def route_after_validate(state: AgentState) -> str:
    """Conditional edge function (used directly by graph.py)."""
    if not state["validation_errors"]:
        return "finalize"
    if state["repair_attempts"] >= settings.max_repair_attempts:
        return "flag_for_review"
    return "repair"


# ---------------------------------------------------------------------------
# Node 6: repair - Reflexion-style self-correction loop
# ---------------------------------------------------------------------------
def repair_node(state: AgentState) -> dict:
    user_prompt = build_repair_user_prompt(
        state["candidate_json"], state["validation_errors"]
    )
    response = _openai_client().chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    candidate = _extract_json(response.choices[0].message.content)
    return {
        "candidate_json": candidate,
        "repair_attempts": state["repair_attempts"] + 1,
    }


# ---------------------------------------------------------------------------
# Node 7 / 8: terminal nodes
# ---------------------------------------------------------------------------
def finalize_node(state: AgentState) -> dict:
    return {
        "final_json": state["candidate_json"],
        "requires_human_review": state["predicted_complexity"] == "Complex",
        "status": "ok",
    }


def flag_for_review_node(state: AgentState) -> dict:
    return {
        "final_json": state["candidate_json"],
        "requires_human_review": True,
        "status": "needs_review",
    }
