"""
API models for opinions.

This module refactors API models to use source-of-truth models from:
- src.llm_extraction.models
- src.neo4j.models 
- src.postgres.models

It uses them directly where possible and extends them where API-specific needs exist.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import date

from src.neo4j.models import Opinion as Neo4jOpinion
from src.llm_extraction.models import (
    OpinionSection,
    CitationTreatment,
    Citation,
    OpinionType,
)

# Re-export models from source of truth for API use
__all__ = [
    "Citation",
    "OpinionSection",
    "CitationTreatment",
    "OpinionType",
    "OpinionBase",
    "OpinionResponse",
    "OpinionDetail",
    "OpinionText",
    "OpinionCitation",
]


class OpinionBase(BaseModel):
    """Base model for opinion data."""

    cluster_id: int = Field(
        ..., description="Unique identifier for the opinion cluster"
    )
    case_name: str = Field(..., description="Name of the case")
    date_filed: date = Field(..., description="Date the opinion was filed")

    @classmethod
    def from_neo4j(cls, opinion: Neo4jOpinion):
        """Create an OpinionBase from a Neo4j Opinion model."""
        return cls(
            cluster_id=opinion.cluster_id,
            case_name=opinion.case_name,
            date_filed=opinion.date_filed,  # use the original date object
        )

    model_config = ConfigDict(from_attributes=True)  # Replaces deprecated orm_mode=True


class OpinionResponse(OpinionBase):
    """Model for opinion summary response."""

    court_id: str = Field(..., description="Court identifier")
    court_name: str = Field(..., description="Name of the court")
    docket_number: Optional[str] = Field(None, description="Docket number")
    citation_count: int = Field(
        0, description="Number of times this opinion has been cited"
    )

    @classmethod
    def from_neo4j(cls, opinion: Neo4jOpinion):
        """Create an OpinionResponse from a Neo4j Opinion model."""
        base = OpinionBase.from_neo4j(opinion)
        return cls(
            **base.model_dump(),
            court_id=str(opinion.court_id),  # Ensure court_id is a string
            court_name=opinion.court_name or "Unknown Court",
            docket_number=opinion.docket_number,
            citation_count=(
                len(opinion.cited_by.all()) if hasattr(opinion, "cited_by") else 0
            ),
        )


class OpinionDetail(OpinionResponse):
    """Model for detailed opinion information."""

    docket_id: Optional[int] = Field(None, description="Docket identifier")
    scdb_votes_majority: Optional[int] = Field(
        None, description="Supreme Court Database majority votes"
    )
    scdb_votes_minority: Optional[int] = Field(
        None, description="Supreme Court Database minority votes"
    )
    brief_summary: Optional[str] = Field(
        None, description="Brief summary of the opinion"
    )
    opinion_type: Optional[OpinionType] = Field(
        None, description="Type of opinion document"
    )

    @classmethod
    def from_neo4j(cls, opinion: Neo4jOpinion):
        """Create an OpinionDetail from a Neo4j Opinion model."""
        base = OpinionResponse.from_neo4j(opinion)
        return cls(
            **base.model_dump(),
            docket_id=opinion.docket_id,
            scdb_votes_majority=opinion.scdb_votes_majority,
            scdb_votes_minority=opinion.scdb_votes_minority,
            brief_summary=opinion.ai_summary,
            opinion_type=opinion.opinion_type,
        )


class OpinionText(BaseModel):
    """Model for opinion text."""

    cluster_id: int = Field(..., description="Opinion cluster identifier")
    text: str = Field(..., description="Full text of the opinion")

    model_config = ConfigDict(from_attributes=True)  # Replaces deprecated orm_mode=True


class OpinionCitation(BaseModel):
    """Model for citation information within an opinion."""

    citing_id: int = Field(..., description="Citing opinion cluster identifier")
    cited_id: int = Field(..., description="Cited opinion cluster identifier")
    citation_text: str = Field(..., description="Original citation text")
    page_number: Optional[int] = Field(
        None, description="Page number where the citation appears"
    )
    treatment: Optional[CitationTreatment] = Field(
        None, description="Citation treatment"
    )
    relevance: Optional[int] = Field(None, description="Relevance score (1-4)")
    reasoning: Optional[str] = Field(None, description="Reasoning for the citation")
    opinion_section: Optional[OpinionSection] = Field(
        None, description="Section of the opinion"
    )

    @classmethod
    def from_neo4j_rel(cls, rel, citing_id: int, cited_id: int):
        """Create an OpinionCitation from a Neo4j CitesRel relationship."""
        return cls(
            citing_id=citing_id,
            cited_id=cited_id,
            citation_text=rel.citation_text or "",
            page_number=rel.page_number,
            treatment=rel.treatment,
            relevance=rel.relevance,
            reasoning=rel.reasoning,
            opinion_section=rel.opinion_section,
        )

    model_config = ConfigDict(from_attributes=True)  # Replaces deprecated orm_mode=True
