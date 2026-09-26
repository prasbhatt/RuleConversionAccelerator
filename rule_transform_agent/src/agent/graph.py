"""
Wires the node functions in nodes.py into a LangGraph StateGraph.

Flow:

    classify --> parse --> retrieve --> generate --> validate --*-- finalize
                                             ^                    |
                                             |                    *-- repair --> validate (loop)
                                             |                    |
                                             +--------------------*-- flag_for_review

`validate` is a conditional branch point (route_after_validate): if the
candidate JSON is clean it goes straight to finalize; if it has errors and
repair attempts remain, it loops back through repair -> validate again
(the Reflexion / self-correction pattern referenced in the thesis
blueprint); if repair attempts are exhausted, the rule is finalized anyway
but flagged for mandatory human review rather than silently returned as if
it were trustworthy.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    classify_node,
    parse_node,
    retrieve_node,
    generate_node,
    validate_node,
    repair_node,
    finalize_node,
    flag_for_review_node,
    route_after_validate,
)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("parse", parse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("repair", repair_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("flag_for_review", flag_for_review_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "parse")
    graph.add_edge("parse", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "finalize": "finalize",
            "repair": "repair",
            "flag_for_review": "flag_for_review",
        },
    )
    graph.add_edge("repair", "validate")
    graph.add_edge("finalize", END)
    graph.add_edge("flag_for_review", END)

    return graph.compile()


def convert_rule(
    source_rule_name: str,
    source_rule_definition: str,
    target_attribute: str | None = None,
    target_locale: str | None = None,
) -> AgentState:
    """Convenience entry point used by cli.py."""
    app = build_graph()
    initial_state: AgentState = {
        "source_rule_name": source_rule_name,
        "source_rule_definition": source_rule_definition,
        "target_attribute": target_attribute,
        "target_locale": target_locale,
        "repair_attempts": 0,
        "validation_errors": [],
    }
    return app.invoke(initial_state)
