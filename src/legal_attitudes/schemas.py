"""Pydantic schemas for structured outputs from legal attitude scales."""
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model with strict config for OpenAI structured output compatibility."""
    model_config = ConfigDict(extra="forbid")


class ObligationResponse(StrictModel):
    """Obligation to Obey the Law scale (8 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")
    q7: int = Field(alias="7")
    q8: int = Field(alias="8")


class ObligationResponseAlt(StrictModel):
    """Obligation to Obey the Law scale (8 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")


class SupportPoliceResponse(StrictModel):
    """Support for Chicago Police scale (6 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")


class SupportCourtsResponse(StrictModel):
    """Support for Chicago Courts scale (6 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")


class PerformanceCourtsResponse(StrictModel):
    """Performance of Chicago Courts scale (10 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")
    q7: int = Field(alias="7")
    q8: int = Field(alias="8")
    q9: int = Field(alias="9")
    q10: int = Field(alias="10")


class PerformancePoliceResponse(StrictModel):
    """Performance of Chicago Police scale (14 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")
    q7: int = Field(alias="7")
    q8: int = Field(alias="8")
    q9: int = Field(alias="9")
    q10: int = Field(alias="10")
    q11: int = Field(alias="11")
    q12: int = Field(alias="12")
    q13: int = Field(alias="13")
    q14: int = Field(alias="14")


class PeersResponse(StrictModel):
    """Peer Disapproval scale (6 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")


class MoralityResponse(StrictModel):
    """Personal Morality scale (6 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")


class DeterrenceResponse(StrictModel):
    """Deterrence/Likelihood of Arrest scale (6 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")


class ComplianceResponse(StrictModel):
    """Self-Reported Compliance scale (6 questions)."""
    q1: int = Field(alias="1")
    q2: int = Field(alias="2")
    q3: int = Field(alias="3")
    q4: int = Field(alias="4")
    q5: int = Field(alias="5")
    q6: int = Field(alias="6")


SCHEMA_REGISTRY = {
    "ObligationResponse": ObligationResponse,
    "ObligationResponseAlt": ObligationResponseAlt,
    "SupportPoliceResponse": SupportPoliceResponse,
    "SupportCourtsResponse": SupportCourtsResponse,
    "PerformanceCourtsResponse": PerformanceCourtsResponse,
    "PerformancePoliceResponse": PerformancePoliceResponse,
    "PeersResponse": PeersResponse,
    "MoralityResponse": MoralityResponse,
    "DeterrenceResponse": DeterrenceResponse,
    "ComplianceResponse": ComplianceResponse,
}


def get_schema(name: str):
    """Look up a Pydantic schema by name."""
    if name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema: {name}. Available: {list(SCHEMA_REGISTRY.keys())}")
    return SCHEMA_REGISTRY[name]
