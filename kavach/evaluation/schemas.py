"""Pydantic contract and wire schemas for the Tier 6 cascade judge."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VerdictEnum(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class CyberRAGEvaluationSchema(BaseModel):
    """`reasoning` is declared first so the audit trace is generated before any score or verdict."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    reasoning: str = Field(
        ...,
        min_length=1,
        description="Claim-by-claim audit of the candidate answer against the retrieved context.",
    )
    faithfulness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Supported claims divided by total claims; 0.0 when a number or identifier was altered.",
    )
    context_relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0.0, 0.5 or 1.0: how much of the needed evidence the retrieved chunks contain.",
    )
    negative_constraint_passed: bool = Field(
        ...,
        description="True when the refusal rule is satisfied or does not apply.",
    )
    conflict_resolution_passed: bool = Field(
        ...,
        description="True when conflicting standard versions were kept separate, or no conflict exists.",
    )
    verdict: VerdictEnum = Field(..., description="PASS, PARTIAL or FAIL.")


FIELD_ORDER: tuple[str, ...] = tuple(CyberRAGEvaluationSchema.model_fields)

_JSON_TYPES: dict[str, dict[str, Any]] = {
    "reasoning": {"type": "string"},
    "faithfulness_score": {"type": "number"},
    "context_relevance_score": {"type": "number"},
    "negative_constraint_passed": {"type": "boolean"},
    "conflict_resolution_passed": {"type": "boolean"},
    "verdict": {"type": "string", "enum": [v.value for v in VerdictEnum]},
}

WIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {name: dict(_JSON_TYPES[name]) for name in FIELD_ORDER},
    "required": list(FIELD_ORDER),
    "additionalProperties": False,
}


def gemini_schema() -> dict[str, Any]:
    """Gemini response schema with explicit property ordering (reasoning first)."""
    properties = {name: {**_JSON_TYPES[name], "type": _JSON_TYPES[name]["type"].upper()} for name in FIELD_ORDER}
    return {
        "type": "OBJECT",
        "properties": properties,
        "required": list(FIELD_ORDER),
        "propertyOrdering": list(FIELD_ORDER),
    }
