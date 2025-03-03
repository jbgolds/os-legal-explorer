from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import StrEnum
from typing import List, Optional, Union, Dict
import json
from json_repair import repair_json
from datetime import datetime
import logging
from src.citation.parser import (
    find_cluster_id,
    clean_citation_text,
    find_cluster_id_fuzzy,
)


logger = logging.getLogger(__name__)

# https://www.law.cornell.edu/citation/6-300
# https://support.vlex.com/how-to/navigate-documents/case-law/treatment-types#undefined
# https://www.lexisnexis.com/pdf/lexis-advance/Shepards-Signal-Indicators-and-analysis-phrases-Final.pdf?srsltid=AfmBOopBSKqOSAAcyYrvQrjavGWoE7ERkWbUrqwM8BrFkYH1RaedBziZ
# https://www.paxton.ai/post/introducing-the-paxton-ai-citator-setting-new-benchmarks-in-legal-research
# from above ^ https://scholarship.law.wm.edu/cgi/viewcontent.cgi?article=1130&context=libpubs; Evaluating Shepard's, KeyCite, and Bcite for Case Validation Accuracy. INACCURATE!

# NOTE to self: Need to think about this model/data in that it's point in time reasoning, Not reasoning today. Need to later combine.


class OpinionSection(StrEnum):
    majority = "MAJORITY"
    concurring = "CONCURRING"
    dissenting = "DISSENTING"

class OpinionType(StrEnum):
    """Type of opinion document."""

    majority = "MAJORITY"
    concurring = "CONCURRING"
    dissenting = "DISSENTING"
    seriatim = "SERIATIM"
    unknown = "UNKNOWN"


class CitationType(StrEnum):
    judicial_opinion = "judicial_opinion"
    statutes_codes_regulations = "statutes_codes_regulations"
    constitution = "constitution"
    administrative_agency_ruling = "administrative_agency_ruling"
    congressional_report = "congressional_report"
    external_submission = "external_submission"
    electronic_resource = "electronic_resource"
    law_review = "law_review"
    legal_dictionary = "legal_dictionary"
    other = "other"


class CitationTreatment(StrEnum):
    POSITIVE = "POSITIVE"  # Only when explicitly relied on as key basis
    NEGATIVE = "NEGATIVE"  # Explicitly disagrees with, distinguishes, or limits
    CAUTION = "CAUTION"  # Expresses doubts or declines to extend
    NEUTRAL = "NEUTRAL"  # Default for background/general reference


class Citation(BaseModel):
    """Model for a single citation extracted from a court opinion."""

    page_number: int = Field(
        ...,
        description="The page number where this citation appears (must be a positive integer).",
    )
    citation_text: str = Field(
        ...,
        description="The entirety of the extracted citation text value, resolved back to the original citation when possible.",
    )
    reasoning: str = Field(
        ...,
        description="Detailed analysis explaining this citation's context and relevance (2-4 sentences).",
    )
    type: CitationType = Field(
        ...,
        description="The citation type (e.g., 'judicial_opinion').",
    )
    treatment: CitationTreatment = Field(
        ...,
        description="The citation treatment (POSITIVE, NEGATIVE, CAUTION, or NEUTRAL).",
    )
    relevance: int = Field(
        ...,
        description="Relevance score between 1 (lowest) and 4 (highest).",
    )

    class Config:
        use_enum_values = True

    # Add method to count citation length
    def count_citation_length(self) -> int:
        """Count the length of all string fields in this citation."""
        return len(self.citation_text) + len(self.reasoning)

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v):
        if isinstance(v, str):
            # Map common input values to enum values
            mapping = {
                "Judicial Opinion": CitationType.judicial_opinion,
                "Statute/Code/Regulation/Rule": CitationType.statutes_codes_regulations,
                "Statute/Code/Regulation": CitationType.statutes_codes_regulations,
                "Constitution": CitationType.constitution,
                "Administrative/Agency Ruling": CitationType.administrative_agency_ruling,
                "Congressional Report": CitationType.congressional_report,
                "External Submission": CitationType.external_submission,
                "Electronic Resource": CitationType.electronic_resource,
                "Law Review": CitationType.law_review,
                "Legal Dictionary": CitationType.legal_dictionary,
                "Other": CitationType.other,
            }

            # Try direct mapping first
            if v in mapping:
                return mapping[v]
            # Try case-insensitive match
            v_lower = v.lower()
            for input_val, enum_val in mapping.items():
                if v_lower == input_val.lower():
                    return enum_val
            # If no match found, try direct enum conversion
            try:
                return CitationType(v)
            except ValueError:
                print(f"Invalid citation type string, defaulting to 'other': {v}")
                return CitationType.other
        elif isinstance(v, CitationType):
            return v
        else:
            print(f"Invalid citation type, defaulting to 'other': {v}")
            return CitationType.other


class CitationAnalysis(BaseModel):
    """Model for a single court opinions complete citation analysis, including the date, brief summary, and citations."""

    date: str = Field(
        ...,
        description="The date of the day the opinion was published, in format YYYY-MM-DD",
    )
    brief_summary: str = Field(
        ..., description="3-5 sentences describing the core holding"
    )
    majority_opinion_citations: List[Citation] = Field(
        default_factory=list,  # Default to empty list instead of requiring it
        description="List of citations from the majority opinion, including footnotes.",
    )
    concurring_opinion_citations: List[Citation] = Field(
        default_factory=list,  # Default to empty list instead of requiring it
        description="List of citations from concurring opinions, including footnotes.",
    )
    dissenting_citations: List[Citation] = Field(
        default_factory=list,  # Default to empty list instead of requiring it
        description="List of citations from dissenting opinions, including footnotes.",
    )

    model_config = ConfigDict(
        json_encoders={str: lambda v: v}  # Preserve Unicode characters in JSON
    )

    # function to count length of all strings in this model
    def count_length(self) -> int:
        sum = 0
        for field in self.model_fields:
            if field.annotation == list or field.annotation == List:
                for item in getattr(self, field.name):
                    if isinstance(item, Citation):
                        sum += item.count_citation_length()
                    # for strings for `notes`
                    elif isinstance(item, str):
                        sum += len(item)
            else:
                sum += len(getattr(self, field.name))
        return sum


class CitationResolved(Citation):
    """
    A resolved citation that can be linked to a document in our database.
    Uses primary_id and primary_table for consistent identification across all document types.
    """

    primary_id: Optional[str] = Field(
        None,
        description="The primary identifier of the resolved document (e.g., cluster_id for opinions)",
    )
    primary_table: Optional[str] = Field(
        None,
        description="The table/source where the primary_id is stored (e.g., 'opinions' for judicial opinions)",
    )
    resolution_confidence: float = Field(
        0.0,
        description="Confidence score for the resolution (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    resolution_method: str = Field(
        "none",
        description="Method used to resolve the citation (exact_match, cleaned_text, fuzzy_match, none)",
    )


# TODO Overlap with CitationAnalysis.
class CombinedResolvedCitationAnalysis(BaseModel):
    date: str
    cluster_id: int
    brief_summary: str
    majority_opinion_citations: list[CitationResolved] = Field(default_factory=list)
    concurring_opinion_citations: list[CitationResolved] = Field(default_factory=list)
    dissenting_citations: list[CitationResolved] = Field(default_factory=list)

    @classmethod
    def from_citations(
        cls,
        citations: list[CitationAnalysis],
        cluster_id: int,
    ) -> "CombinedResolvedCitationAnalysis":

        return cls(
            date=citations[0].date,
            cluster_id=cluster_id,
            brief_summary=citations[0].brief_summary,
            majority_opinion_citations=[
                resolve_citation(citation)
                for citation_analysis in citations
                for citation in citation_analysis.majority_opinion_citations
            ],
            concurring_opinion_citations=[
                resolve_citation(citation)
                for citation_analysis in citations
                for citation in citation_analysis.concurring_opinion_citations
            ],
            dissenting_citations=[
                resolve_citation(citation)
                for citation_analysis in citations
                for citation in citation_analysis.dissenting_citations
            ],
        )

   


def _resolve_opinion_citation(citation: Citation) -> CitationResolved:
    """
    Convert a Citation to a CitationResolved with a postgres db lookup.

    This improved version adds a confidence score and handles more citation formats.
    """
    
    if citation.type != CitationType.judicial_opinion:
            raise ValueError("Cannot use this function to resolve non judicial opinions.")

    
    # Initialize with default values
    resolved_cluster_id = None
    confidence_score = 0.0
    resolution_method = "none"
    
    # Try exact citation lookup first
    resolved_cluster_id = find_cluster_id(citation.citation_text)

    if resolved_cluster_id:
        confidence_score = 1.0
        resolution_method = "exact_match"
    else:
        # Try alternative resolution methods if exact match fails
        # 1. Try with cleaned citation text (remove extra spaces, normalize punctuation)
        cleaned_text = clean_citation_text(citation.citation_text)
        if cleaned_text != citation.citation_text:
            resolved_cluster_id = find_cluster_id(cleaned_text)
            if resolved_cluster_id:
                confidence_score = 0.9
                resolution_method = "cleaned_text"

        # 2. Try fuzzy matching if still not resolved
        if not resolved_cluster_id and len(citation.citation_text) > 10:
            resolved_cluster_id, match_score = find_cluster_id_fuzzy(
                citation.citation_text
            )
            if resolved_cluster_id:
                confidence_score = min(0.7, match_score)  # Cap at 0.7
                resolution_method = "fuzzy_match"

    # Create the resolved citation with additional metadata
    return CitationResolved(
        **citation.model_dump(),  # Copy all existing fields
        primary_id=resolved_cluster_id,
        primary_table="opinion_cluster",
        resolution_confidence=confidence_score,
        resolution_method=resolution_method,
    )


def resolve_citation(citation: Citation) -> CitationResolved:
    if citation.type == CitationType.judicial_opinion:
        return _resolve_opinion_citation(citation)
    else:
        return CitationResolved(
        **citation.model_dump(),  # Copy all existing fields
        primary_id=None,
        primary_table=None,
        resolution_confidence=0.0,
        resolution_method="none",
    )
