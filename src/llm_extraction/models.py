from argparse import Action
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import StrEnum
from typing import List, Optional, TYPE_CHECKING, ForwardRef, Union, Dict
import json
from json_repair import repair_json
from src.postgres.database import (
    find_cluster_id,
    get_db_session,
    Citation as DBCitation,
)
from datetime import datetime
import logging

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from src.postgres.models import CitationExtraction, OpinionClusterExtraction
else:
    # Forward references for type hints
    CitationExtraction = ForwardRef("CitationExtraction")
    OpinionClusterExtraction = ForwardRef("OpinionClusterExtraction")

logger = logging.getLogger(__name__)

# https://www.law.cornell.edu/citation/6-300
# https://support.vlex.com/how-to/navigate-documents/case-law/treatment-types#undefined
# https://www.lexisnexis.com/pdf/lexis-advance/Shepards-Signal-Indicators-and-analysis-phrases-Final.pdf?srsltid=AfmBOopBSKqOSAAcyYrvQrjavGWoE7ERkWbUrqwM8BrFkYH1RaedBziZ
# https://www.paxton.ai/post/introducing-the-paxton-ai-citator-setting-new-benchmarks-in-legal-research
# from above ^ https://scholarship.law.wm.edu/cgi/viewcontent.cgi?article=1130&context=libpubs; Evaluating Shepard's, KeyCite, and Bcite for Case Validation Accuracy. INACCURATE!

# NOTE to self: Need to think about this model/data in that it's point in time reasoning, Not reasoning today. Need to later combine.


signals = """
(a) Signals that indicate support:
- E.g., : Authority states the proposition with which the citation is associated. Other authorities, not cited, do as well «e.g.». "E.g." used with other signals (in which case it is preceded by a comma) similarly indicates the existence of other authorities not cited.
- Accord: Used following citation to authority referred to in text when there are additional authorities that either state or clearly support the proposition with which the citation is associated, but the text quotes only one. Similarly, the law of one jurisdiction may be cited as being in accord with that of another «e.g.».
- See: Authority supports the proposition with which the citation is associated either implicitly or in the form of dicta «e.g.» .
- See also: Authority is additional support for the proposition with which the citation is associated (but less direct than that indicated by "see" or "accord"). "See also" is commonly used to refer readers to authorities already cited or discussed «e.g.». Generally, it is helpful to include a parenthetical explanation of the source material's relevance following a citation introduced by "see also."
- Cf. : Authority supports by analogy. "Cf." literally means "compare." The citation will only appear relevant to the reader if it is explained. Consequently, in most cases a parenthetical explanations of the analogy should be included «e.g.».
- Followed by: Authority supports the proposition with which the citation is associated.

(b) Signals that suggest a useful comparison.
- Compare ... with ... : Comparison of authorities that supports proposition. Either side of the comparison can have more than one item linked with "and" «e.g.». Generally, a parenthetical explanation of the comparison should be furnished.

(c) Signals that indicate contradiction.
- Contra: Authority directly states the contrary of the proposition with which the citation is associated «e.g.».
- But see: Authority clearly supports the contrary of the proposition with which citation is associated «e.g.».
- But cf.: Authority supports the contrary of the position with which the citation is associated by analogy. In most cases a parenthetical explanations of the analogy should be furnished. The word "but" is omitted from the signal when it follows another negative signal «e.g.».
- Overruled by: The citing case expressly overrules or disapproves all or part of the case cited.
- Abrogated by: The citing case effectively, but not explicitly, overrules or departs from the case cited.
- Superseded by: The citing reference—typically a session law, other record of legislative action or a record of administrative action— supersedes the statute, regulation or order cited.

(d) Signals that indicate background material.
- See generally: Authority presents useful background. Parenthetical explanations of the source materials' relevance are generally useful «e.g.».

(e) Combining a signal with "e.g."
- E.g.,: In addition to the cited authority, there are numerous others that state, support, or contradict the proposition (with the other signal indicating which) but citation to them would not be helpful or necessary. The preceding signal is separated from "e.g." by a comma «e.g.».
"""


class OpinionSection(StrEnum):
    majority = "MAJORITY"
    concurring = "CONCURRING"
    dissenting = "DISSENTING"


class OpinionType(StrEnum):
    """Type of opinion document."""

    majority = "MAJORITY"
    concurring = "CONCURRING"
    dissenting = "DISSENTING"
    per_curiam = "PER_CURIAM"
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
    positive = "POSITIVE"  # Only when explicitly relied on as key basis
    negative = "NEGATIVE"  # Explicitly disagrees with, distinguishes, or limits
    caution = "CAUTION"  # Expresses doubts or declines to extend
    neutral = "NEUTRAL"  # Default for background/general reference


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
        ...,
        description="List of citations from the majority opinion, including footnotes.",
    )
    concurring_opinion_citations: List[Citation] = Field(
        ...,
        description="List of citations from concurring opinions, including footnotes.",
    )
    dissenting_citations: List[Citation] = Field(
        ...,
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
    """This is for the resolved citation, which includes the cluster_id of the resolved citation for case law;"""

    resolved_opinion_cluster: int | None = Field(
        None,  # Changed from ... to None to make it optional
        description="The cluster_id of the resolved citation for case law; not for statutes or regulations.",
    )
    resolved_statute_code_regulation_rule: int | None = Field(
        None,  # Changed from ... to None to make it optional
        description="TBD, NEED TO CREATE THIS DATABASE TABLE",
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


def resolve_citation(citation: Citation) -> CitationResolved:
    """
    Convert a Citation to a CitationResolved with a postgres db lookup.

    This improved version adds a confidence score and handles more citation formats.
    """
    # Import the consolidated citation parser functions
    from src.citation.parser import (
        find_cluster_id,
        clean_citation_text,
        find_cluster_id_fuzzy,
    )

    # Initialize with default values
    resolved_cluster_id = None
    confidence_score = 0.0
    resolution_method = "none"

    if citation.type == CitationType.judicial_opinion:
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
        resolved_opinion_cluster=resolved_cluster_id,
        resolved_statute_code_regulation_rule=None,
        resolution_confidence=confidence_score,
        resolution_method=resolution_method,
    )


def get_reporter_mappings() -> dict:
    """
    Get mappings between canonical reporter names and their variations.

    This function is now a wrapper around the consolidated citation parser.

    Returns:
        Dictionary mapping canonical reporter names to lists of variations
    """
    # Import the consolidated citation parser
    from src.citation.parser import (
        get_reporter_mappings as consolidated_get_reporter_mappings,
    )

    # Use the consolidated function
    return consolidated_get_reporter_mappings()


def create_reporter_lookup(reporter_mappings: dict) -> dict:
    """
    Create a reverse lookup dictionary for reporter variations.

    This function is now a wrapper around the consolidated citation parser.

    Args:
        reporter_mappings: Dictionary mapping canonical names to variations

    Returns:
        Dictionary mapping variations to canonical names
    """
    # Import the consolidated citation parser
    from src.citation.parser import (
        create_reporter_lookup as consolidated_create_reporter_lookup,
    )

    # Use the consolidated function
    return consolidated_create_reporter_lookup(reporter_mappings)


def normalize_reporter(reporter_raw: str, reporter_lookup: dict) -> str:
    """
    Normalize reporter abbreviation to canonical form.

    This function is now a wrapper around the consolidated citation parser.

    Args:
        reporter_raw: Raw reporter string
        reporter_lookup: Dictionary mapping variations to canonical names

    Returns:
        Normalized reporter string
    """
    # Import the consolidated citation parser
    from src.citation.parser import (
        normalize_reporter as consolidated_normalize_reporter,
    )

    # Use the consolidated function
    return consolidated_normalize_reporter(reporter_raw, reporter_lookup)


def extract_citation_components(citation_text: str) -> list:
    """
    Extract citation components (volume, reporter, page) from citation text.

    This function is now a wrapper around the consolidated citation parser.

    Args:
        citation_text: Citation text to parse

    Returns:
        List of dictionaries containing extracted components
    """
    # Import the consolidated citation parser
    from src.citation.parser import (
        extract_citation_components as consolidated_extract_citation_components,
    )

    # Use the consolidated function
    return consolidated_extract_citation_components(citation_text)


def extract_parallel_citations(citation_text: str) -> list:
    """
    Extract components from parallel citations.

    This function is now a wrapper around the consolidated citation parser.

    Args:
        citation_text: Citation text to parse

    Returns:
        List of dictionaries containing extracted components
    """
    # Import the consolidated citation parser
    from src.citation.parser import (
        extract_parallel_citations as consolidated_extract_parallel_citations,
    )

    # Use the consolidated function
    return consolidated_extract_parallel_citations(citation_text)


# TODO Overlap with CitationAnalysis.
class CombinedResolvedCitationAnalysis(BaseModel):
    date: str
    cluster_id: int
    brief_summary: str
    majority_opinion_citations: list[CitationResolved]
    concurring_opinion_citations: list[
        CitationResolved
    ]  # Changed from concurrent to concurring for consistency
    dissenting_citations: list[CitationResolved]

    @classmethod
    def from_citations_json(
        cls, citations: Union[str, Dict, CitationAnalysis, List], cluster_id: int
    ) -> Optional["CombinedResolvedCitationAnalysis"]:
        """
        Create a CombinedResolvedCitationAnalysis from a CitationAnalysis object or its JSON representation.

        This is a simplified wrapper around from_citations that handles JSON parsing.
        For complex transformations, use from_citations directly with parsed CitationAnalysis objects.

        Args:
            citations: CitationAnalysis object, its JSON string, dict representation, or list
            cluster_id: The cluster ID for the citing opinion

        Returns:
            CombinedResolvedCitationAnalysis or None if parsing fails
        """
        # Skip processing if cluster_id is invalid
        if not cluster_id or cluster_id == -999999 or cluster_id < 0:
            logger.warning(f"Invalid cluster ID: {cluster_id}, skipping")
            return None

        try:
            # If already a CitationAnalysis, use directly
            if isinstance(citations, CitationAnalysis):
                return cls.from_citations([citations], cluster_id)

            # If it's a list of CitationAnalysis objects
            if isinstance(citations, list) and all(
                isinstance(item, CitationAnalysis) for item in citations
            ):
                return cls.from_citations(citations, cluster_id)

            # If string, try to parse as JSON
            if isinstance(citations, str):
                try:
                    citation_data = json.loads(citations)
                except json.JSONDecodeError as e:
                    # Try to repair the JSON before giving up
                    logger.warning(
                        f"JSON parse error for cluster {cluster_id}: {str(e)}, attempting repair"
                    )
                    try:
                        repaired = repair_json(citations)
                        citation_data = json.loads(repaired)
                        logger.info(
                            f"Successfully repaired JSON for cluster {cluster_id}"
                        )
                    except Exception as repair_error:
                        logger.error(
                            f"JSON repair failed for cluster {cluster_id}: {str(repair_error)}"
                        )
                        return None
            else:
                citation_data = citations

            # Convert to CitationAnalysis objects
            citation_analyses = []

            # Handle dictionary input
            if isinstance(citation_data, dict):
                try:
                    citation_analysis = CitationAnalysis(
                        date=citation_data.get(
                            "date", datetime.date.today().isoformat()
                        ),
                        brief_summary=citation_data.get(
                            "brief_summary", "No summary available"
                        ),
                        majority_opinion_citations=citation_data.get(
                            "majority_opinion_citations", []
                        ),
                        concurring_opinion_citations=citation_data.get(
                            "concurring_opinion_citations", []
                        ),
                        dissenting_citations=citation_data.get(
                            "dissenting_citations", []
                        ),
                    )
                    citation_analyses.append(citation_analysis)
                except Exception as e:
                    logger.error(
                        f"Failed to process citation dictionary for cluster {cluster_id}: {str(e)}"
                    )
                    return None

            # Handle list input
            elif isinstance(citation_data, list):
                for item in citation_data:
                    if isinstance(item, dict):
                        try:
                            citation_analysis = CitationAnalysis(
                                date=item.get(
                                    "date", datetime.date.today().isoformat()
                                ),
                                brief_summary=item.get(
                                    "brief_summary", "No summary available"
                                ),
                                majority_opinion_citations=item.get(
                                    "majority_opinion_citations", []
                                ),
                                concurring_opinion_citations=item.get(
                                    "concurring_opinion_citations", []
                                ),
                                dissenting_citations=item.get(
                                    "dissenting_citations", []
                                ),
                            )
                            citation_analyses.append(citation_analysis)
                        except Exception as e:
                            logger.warning(
                                f"Failed to process citation dict at index {citation_data.index(item)}: {str(e)}"
                            )

            if citation_analyses:
                return cls.from_citations(citation_analyses, cluster_id)

            logger.error(f"Could not process citation data for cluster {cluster_id}")
            return None

        except Exception as e:
            logger.error(
                f"Failed to create CitationAnalysis for cluster {cluster_id}: {e}"
            )
            return None

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

    def get_opinion_nodes(self) -> dict:
        """Return a dictionary of unique opinion nodes needed for this citation analysis.
        The key is the opinion cluster_id and the value is a dict of properties.
        For the citing opinion, includes 'date_filed'; for others, only 'cluster_id'."""
        nodes = {}
        # Add the citing opinion node
        nodes[self.cluster_id] = {
            "cluster_id": self.cluster_id,
            "date_filed": self.date,
            "type": "judicial_opinion",
        }

        # Helper function to process a list of citations
        def process_citations(citations):
            for citation in citations:
                if citation.resolved_opinion_cluster is not None:
                    if citation.resolved_opinion_cluster not in nodes:
                        nodes[citation.resolved_opinion_cluster] = {
                            "cluster_id": citation.resolved_opinion_cluster,
                            "type": citation.type,  # Add citation type from the model
                        }

        process_citations(self.majority_opinion_citations)
        process_citations(self.concurring_opinion_citations)
        process_citations(self.dissenting_citations)

        return nodes


def to_sql_models(
    combined: CombinedResolvedCitationAnalysis,
) -> tuple[OpinionClusterExtraction, list[CitationExtraction]]:
    """Convert a CombinedResolvedCitationAnalysis to SQL models.

    Returns:
        Tuple containing:
        - OpinionClusterExtraction instance
        - List of CitationExtraction instances
    """
    # Create the opinion cluster extraction
    cluster = OpinionClusterExtraction(
        cluster_id=combined.cluster_id,
        date_filed=datetime.strptime(combined.date, "%Y-%m-%d"),
        brief_summary=combined.brief_summary,
    )

    citations = []

    # Helper to process citations from a section
    def process_section_citations(
        citation_list: list[CitationResolved], section: OpinionSection
    ):
        for citation in citation_list:
            cite = CitationExtraction(
                opinion_cluster_extraction_id=cluster.id,  # This will be set after DB insert
                section=section,
                citation_type=citation.type,
                citation_text=citation.citation_text,
                page_number=citation.page_number,
                treatment=citation.treatment,
                relevance=citation.relevance,
                reasoning=citation.reasoning,
                resolved_opinion_cluster=citation.resolved_opinion_cluster,
                resolved_text=citation.citation_text,  # Using original text as resolved for now
            )
            citations.append(cite)

    # Process each section
    process_section_citations(
        combined.majority_opinion_citations, OpinionSection.majority
    )
    process_section_citations(
        combined.concurring_opinion_citations, OpinionSection.concurring
    )
    process_section_citations(combined.dissenting_citations, OpinionSection.dissenting)

    return cluster, citations


def from_sql_models(
    cluster: OpinionClusterExtraction,
) -> CombinedResolvedCitationAnalysis:
    """Convert SQL models back to a CombinedResolvedCitationAnalysis.

    Args:
        cluster: OpinionClusterExtraction instance with citations relationship loaded
    """
    # Sort citations by section
    majority_citations = []
    concurring_citations = []
    dissenting_citations = []

    for citation in cluster.citations:
        resolved_citation = CitationResolved(
            page_number=citation.page_number or 1,  # Default to 1 if None
            citation_text=citation.citation_text,
            reasoning=citation.reasoning or "",  # Default to empty string if None
            type=citation.citation_type,
            treatment=citation.treatment
            or CitationTreatment.neutral,  # Default to neutral if None
            relevance=citation.relevance or 1,  # Default to 1 if None
            resolved_opinion_cluster=citation.resolved_opinion_cluster,
            resolved_statute_code_regulation_rule=None,  # Not implemented yet
        )

        if citation.section == OpinionSection.majority:
            majority_citations.append(resolved_citation)
        elif citation.section == OpinionSection.concurring:
            concurring_citations.append(resolved_citation)
        elif citation.section == OpinionSection.dissenting:
            dissenting_citations.append(resolved_citation)

    return CombinedResolvedCitationAnalysis(
        date=cluster.date_filed.strftime("%Y-%m-%d"),
        cluster_id=cluster.cluster_id,
        brief_summary=cluster.brief_summary,
        majority_opinion_citations=majority_citations,
        concurring_opinion_citations=concurring_citations,
        dissenting_citations=dissenting_citations,
    )
