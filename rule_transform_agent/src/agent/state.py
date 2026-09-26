"""
Shared state object that flows through every node of the LangGraph agent.
Every node reads from and writes to this one structure - see thesis Section
"Ordered-program execution model" for why an explicit, typed, inspectable
state was preferred over letting the LLM freely improvise its own workflow.
"""
from __future__ import annotations
from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    # ---- input ----
    source_rule_name: str
    source_rule_definition: str
    target_attribute: Optional[str]
    target_locale: Optional[str]

    # ---- produced by classify_node ----
    predicted_category: str
    predicted_complexity: str

    # ---- produced by parse_node ----
    draft_statements: list
    referenced_attributes: list[str]

    # ---- produced by retrieve_node ----
    retrieved_examples: list[dict]

    # ---- produced by generate_node / repair_node ----
    candidate_json: Optional[dict]
    repair_attempts: int
    validation_errors: list[str]

    # ---- final ----
    final_json: Optional[dict]
    requires_human_review: bool
    status: str  # "ok" | "needs_review" | "failed"
