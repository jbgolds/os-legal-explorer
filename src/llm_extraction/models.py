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


def clean_citation_text(text: str) -> str:
    """
    Clean citation text by normalizing spaces, punctuation, and case.

    Args:
        text: Original citation text

    Returns:
        Cleaned citation text
    """
    import re

    # Remove extra whitespace
    cleaned = re.sub(r"\s+", " ", text.strip())

    # Normalize common citation patterns
    cleaned = re.sub(r"(\d+)\s*U\.\s*S\.\s*(\d+)", r"\1 U.S. \2", cleaned)
    cleaned = re.sub(r"(\d+)\s*S\.\s*Ct\.\s*(\d+)", r"\1 S.Ct. \2", cleaned)
    cleaned = re.sub(r"(\d+)\s*F\.\s*(\d+)\s*d\s*\.?\s*(\d+)", r"\1 F.\2d \3", cleaned)

    return cleaned


def find_cluster_id_fuzzy(
    citation_text: str, threshold: float = 0.8
) -> tuple[int | None, float]:
    """
    Find cluster_id using fuzzy matching for citations that don't match exactly.

    This implementation uses multiple strategies to find the best match:
    1. Extract key components (volume, reporter, page) with regex patterns
    2. Try partial matching with extracted components
    3. Use fuzzy string matching for reporter abbreviations
    4. Score matches based on component similarity

    Args:
        citation_text: Citation text to look up
        threshold: Minimum similarity score (0-1) to consider a match

    Returns:
        Tuple of (cluster_id, confidence_score) if found, (None, 0.0) otherwise
    """
    import re
    from sqlalchemy import or_, and_

    logger.info(f"Attempting fuzzy match for: {citation_text}")

    # Define reporter mappings once at the module level to avoid recreating it on each call
    reporter_mappings = get_reporter_mappings()

    # Create reverse mapping for lookup
    reporter_lookup = create_reporter_lookup(reporter_mappings)

    # Extract components from citation text
    extracted_components = extract_citation_components(citation_text)

    if not extracted_components:
        logger.warning(f"Could not extract components from citation: {citation_text}")
        return None, 0.0

    # Try to find matches in the database
    try:
        return find_best_match(
            extracted_components,
            reporter_mappings,
            reporter_lookup,
            threshold,
            citation_text,
        )
    except Exception as e:
        logger.error(
            f"Error in fuzzy citation matching for '{citation_text}': {str(e)}"
        )
        return None, 0.0


def get_reporter_mappings() -> dict:
    """
    Get mappings of reporter abbreviations to their variations.

    Returns:
        Dictionary mapping canonical reporter names to lists of variations
    """
    return {
        "U.S.": ["U.S.", "U. S.", "US", "U S", "U.S", "United States Reports"],
        "S.Ct.": [
            "S.Ct.",
            "S. Ct.",
            "S.Ct",
            "S Ct",
            "Sup. Ct.",
            "Supreme Court Reporter",
            "S. Court",
            "Sup. Court",
        ],
        "F.": ["F.", "F", "Fed.", "Federal Reporter"],
        "F.2d": [
            "F.2d",
            "F. 2d",
            "F2d",
            "F 2d",
            "Fed. 2d",
            "Federal Reporter, Second Series",
            "Fed. Rep. 2d",
        ],
        "F.3d": [
            "F.3d",
            "F. 3d",
            "F3d",
            "F 3d",
            "Fed. 3d",
            "Federal Reporter, Third Series",
            "Fed. Rep. 3d",
        ],
        "F.Supp.": [
            "F.Supp.",
            "F. Supp.",
            "F Supp",
            "Fed. Supp.",
            "Federal Supplement",
        ],
        "F.Supp.2d": [
            "F.Supp.2d",
            "F. Supp. 2d",
            "F Supp 2d",
            "Fed. Supp. 2d",
            "Federal Supplement, Second Series",
        ],
        "F.Supp.3d": [
            "F.Supp.3d",
            "F. Supp. 3d",
            "F Supp 3d",
            "Fed. Supp. 3d",
            "Federal Supplement, Third Series",
        ],
        "L.Ed.": ["L.Ed.", "L. Ed.", "L Ed", "Lawyers Edition"],
        "L.Ed.2d": [
            "L.Ed.2d",
            "L. Ed. 2d",
            "L Ed 2d",
            "Lawyers Edition, Second Series",
        ],
        "Cal.": ["Cal.", "Cal", "California Reports"],
        "Cal.2d": ["Cal.2d", "Cal. 2d", "Cal 2d", "California Reports, Second Series"],
        "Cal.3d": ["Cal.3d", "Cal. 3d", "Cal 3d", "California Reports, Third Series"],
        "Cal.4th": [
            "Cal.4th",
            "Cal. 4th",
            "Cal 4th",
            "California Reports, Fourth Series",
        ],
        "Cal.5th": [
            "Cal.5th",
            "Cal. 5th",
            "Cal 5th",
            "California Reports, Fifth Series",
        ],
        "N.Y.": ["N.Y.", "N. Y.", "NY", "New York Reports"],
        "N.Y.2d": ["N.Y.2d", "N. Y. 2d", "NY 2d", "New York Reports, Second Series"],
        "N.Y.3d": ["N.Y.3d", "N. Y. 3d", "NY 3d", "New York Reports, Third Series"],
        "A.": ["A.", "A", "Atlantic Reporter"],
        "A.2d": ["A.2d", "A. 2d", "A2d", "Atlantic Reporter, Second Series"],
        "A.3d": ["A.3d", "A. 3d", "A3d", "Atlantic Reporter, Third Series"],
        "P.": ["P.", "P", "Pacific Reporter"],
        "P.2d": ["P.2d", "P. 2d", "P2d", "Pacific Reporter, Second Series"],
        "P.3d": ["P.3d", "P. 3d", "P3d", "Pacific Reporter, Third Series"],
        "N.E.": ["N.E.", "N. E.", "NE", "North Eastern Reporter"],
        "N.E.2d": [
            "N.E.2d",
            "N. E. 2d",
            "NE 2d",
            "North Eastern Reporter, Second Series",
        ],
        "N.E.3d": [
            "N.E.3d",
            "N. E. 3d",
            "NE 3d",
            "North Eastern Reporter, Third Series",
        ],
        "N.W.": ["N.W.", "N. W.", "NW", "North Western Reporter"],
        "N.W.2d": [
            "N.W.2d",
            "N. W. 2d",
            "NW 2d",
            "North Western Reporter, Second Series",
        ],
        "S.E.": ["S.E.", "S. E.", "SE", "South Eastern Reporter"],
        "S.E.2d": [
            "S.E.2d",
            "S. E. 2d",
            "SE 2d",
            "South Eastern Reporter, Second Series",
        ],
        "S.W.": ["S.W.", "S. W.", "SW", "South Western Reporter"],
        "S.W.2d": [
            "S.W.2d",
            "S. W. 2d",
            "SW 2d",
            "South Western Reporter, Second Series",
        ],
        "S.W.3d": [
            "S.W.3d",
            "S. W. 3d",
            "SW 3d",
            "South Western Reporter, Third Series",
        ],
    }


def create_reporter_lookup(reporter_mappings: dict) -> dict:
    """
    Create a reverse mapping from reporter variations to canonical names.

    Args:
        reporter_mappings: Dictionary mapping canonical reporter names to lists of variations

    Returns:
        Dictionary mapping each variation to its canonical name
    """
    reporter_lookup = {}
    for canonical, variations in reporter_mappings.items():
        for var in variations:
            reporter_lookup[var] = canonical
    return reporter_lookup


def extract_citation_components(citation_text: str) -> list:
    """
    Extract citation components (volume, reporter, page) from citation text.

    Args:
        citation_text: Citation text to parse

    Returns:
        List of dictionaries containing extracted components
    """
    import re

    # Define regex patterns for different citation formats
    patterns = [
        # Standard reporter pattern: volume reporter page
        r"(\d+)\s+([A-Za-z\.\s]+)\s+(\d+)",
        # Pattern with reporter in the middle: volume reporter page
        r"(\d+)\s+([A-Za-z\.]+(?:\s+[A-Za-z\.]+)?(?:\s+\d+)?)\s+(\d+)",
        # Pattern with parenthetical year: reporter volume, page (year)
        r"([A-Za-z\.]+)\s+(\d+),\s+(\d+)\s*\(\d{4}\)",
        # Pattern with reporter at beginning: reporter volume, page
        r"([A-Za-z\.]+(?:\s+[A-Za-z\.]+)?)\s+(\d+),\s+(\d+)",
        # Pattern with series number in reporter: volume reporter series page
        r"(\d+)\s+([A-Za-z\.]+)\s+(\d+)[dhnrs]+\s+(\d+)",
        # Pattern for parallel citations: volume1 reporter1 page1, volume2 reporter2 page2
        r"(\d+)\s+([A-Za-z\.]+(?:\s+[A-Za-z\.]+)?)\s+(\d+)(?:\s*,\s*\d+\s+[A-Za-z\.]+(?:\s+[A-Za-z\.]+)?\s+\d+)?",
        # Pattern for citations with section symbol: § volume-page
        r"§\s*(\d+)-(\d+)",
    ]

    extracted_components = []

    # Try each pattern to extract components
    for pattern in patterns:
        matches = re.search(pattern, citation_text)
        if matches:
            # Different patterns have different group orders
            if (
                pattern == patterns[0]
                or pattern == patterns[1]
                or pattern == patterns[5]
            ):
                volume, reporter, page = matches.groups()
            elif pattern == patterns[2] or pattern == patterns[3]:
                reporter, volume, page = matches.groups()
            elif pattern == patterns[4]:  # Pattern with series in reporter
                volume, reporter_base, series, page = matches.groups()
                reporter = f"{reporter_base} {series}"
            elif pattern == patterns[6]:  # Section symbol pattern
                volume, page = matches.groups()
                reporter = "U.S.C."  # Assume U.S. Code for section symbols

            # Clean up the reporter string
            if reporter:
                reporter = reporter.strip()

            # Add to extracted components
            extracted_components.append(
                {
                    "volume": volume,
                    "reporter": reporter,
                    "page": page,
                    "pattern": pattern,
                }
            )

    # Special handling for parallel citations (multiple citations for the same case)
    if not extracted_components:
        extracted_components = extract_parallel_citations(citation_text)

    return extracted_components


def extract_parallel_citations(citation_text: str) -> list:
    """
    Extract components from parallel citations.

    Args:
        citation_text: Citation text to parse

    Returns:
        List of dictionaries containing extracted components
    """
    import re

    extracted_components = []

    # Try to extract parallel citations
    parallel_pattern = (
        r"(\d+)\s+([A-Za-z\.\s]+)\s+(\d+)\s*,\s*(\d+)\s+([A-Za-z\.\s]+)\s+(\d+)"
    )
    parallel_matches = re.search(parallel_pattern, citation_text)

    if parallel_matches:
        vol1, rep1, page1, vol2, rep2, page2 = parallel_matches.groups()

        # Add both citations
        extracted_components.append(
            {
                "volume": vol1,
                "reporter": rep1.strip(),
                "page": page1,
                "pattern": "parallel_first",
            }
        )

        extracted_components.append(
            {
                "volume": vol2,
                "reporter": rep2.strip(),
                "page": page2,
                "pattern": "parallel_second",
            }
        )

        logger.info(
            f"Extracted parallel citations: {vol1} {rep1} {page1} and {vol2} {rep2} {page2}"
        )

    return extracted_components


def normalize_reporter(reporter_raw: str, reporter_lookup: dict) -> str:
    """
    Normalize reporter abbreviation to canonical form.

    Args:
        reporter_raw: Raw reporter abbreviation
        reporter_lookup: Dictionary mapping variations to canonical names

    Returns:
        Normalized reporter abbreviation
    """
    # Try direct lookup
    reporter = reporter_lookup.get(reporter_raw, reporter_raw)

    # Try case-insensitive lookup if no direct match
    if reporter == reporter_raw:
        reporter_raw_lower = reporter_raw.lower()
        for key, value in reporter_lookup.items():
            if key.lower() == reporter_raw_lower:
                reporter = reporter_lookup[key]
                logger.info(
                    f"Found case-insensitive match for reporter: '{reporter_raw}' -> '{reporter}'"
                )
                break

    return reporter


def find_best_match(
    extracted_components: list,
    reporter_mappings: dict,
    reporter_lookup: dict,
    threshold: float,
    citation_text: str,
) -> tuple[int | None, float]:
    """
    Find the best matching citation in the database.

    Args:
        extracted_components: List of dictionaries containing extracted components
        reporter_mappings: Dictionary mapping canonical reporter names to lists of variations
        reporter_lookup: Dictionary mapping variations to canonical names
        threshold: Minimum similarity score to consider a match
        citation_text: Original citation text for logging

    Returns:
        Tuple of (cluster_id, confidence_score) if found, (None, 0.0) otherwise
    """
    from sqlalchemy import and_

    best_match = None
    best_score = 0.0

    with get_db_session() as session:
        # Try each extracted component
        for component in extracted_components:
            volume = component["volume"]
            reporter_raw = component["reporter"]
            page = component["page"]

            # Normalize reporter
            reporter = normalize_reporter(reporter_raw, reporter_lookup)

            # Try exact match first
            match, score = try_exact_match(session, volume, reporter, page)
            if score > best_score:
                best_match, best_score = match, score

            # If no good match, try fuzzy reporter matching
            if best_score < threshold:
                match, score = try_fuzzy_reporter_match(
                    session, volume, reporter, page, reporter_mappings
                )
                if score > best_score:
                    best_match, best_score = match, score

            # If still no good match, try just volume and page
            if best_score < threshold:
                match, score = try_volume_page_match(session, volume, page)
                if score > best_score:
                    best_match, best_score = match, score

        # If still no good match, try alternative approaches
        if best_score < threshold:
            match, score = try_simplified_reporter_match(session, extracted_components)
            if score > best_score:
                best_match, best_score = match, score

        # Return the best match if it meets the threshold
        if best_match and best_score >= threshold:
            logger.info(
                f"Found fuzzy match for '{citation_text}': cluster_id={best_match.cluster_id}, score={best_score}"
            )
            return best_match.cluster_id, best_score

        logger.warning(
            f"No fuzzy match found for '{citation_text}' that meets threshold {threshold}"
        )
        return None, 0.0


def try_exact_match(
    session, volume: str, reporter: str, page: str
) -> tuple[object | None, float]:
    """
    Try to find an exact match for volume, reporter, and page.

    Args:
        session: Database session
        volume: Citation volume
        reporter: Citation reporter
        page: Citation page

    Returns:
        Tuple of (match, score) if found, (None, 0.0) otherwise
    """
    from sqlalchemy import and_

    best_match = None
    best_score = 0.0

    if volume and page:
        query = session.query(DBCitation).filter(
            and_(DBCitation.volume == volume, DBCitation.page == page)
        )

        if reporter:
            exact_matches = query.filter(DBCitation.reporter == reporter).all()
            if exact_matches:
                # Found exact matches, score them highly
                for match in exact_matches:
                    score = 0.95  # High confidence for volume+reporter+page match
                    if score > best_score:
                        best_match = match
                        best_score = score

    return best_match, best_score


def try_fuzzy_reporter_match(
    session, volume: str, reporter: str, page: str, reporter_mappings: dict
) -> tuple[object | None, float]:
    """
    Try to find a match using fuzzy reporter matching.

    Args:
        session: Database session
        volume: Citation volume
        reporter: Citation reporter
        page: Citation page
        reporter_mappings: Dictionary mapping canonical reporter names to lists of variations

    Returns:
        Tuple of (match, score) if found, (None, 0.0) otherwise
    """
    from sqlalchemy import and_

    best_match = None
    best_score = 0.0

    if volume and page and reporter:
        # Get all possible variations of this reporter
        reporter_variations = []
        for canonical, variations in reporter_mappings.items():
            if reporter in variations or reporter == canonical:
                reporter_variations.extend(variations)

        if reporter_variations:
            query = session.query(DBCitation).filter(
                and_(DBCitation.volume == volume, DBCitation.page == page)
            )

            fuzzy_matches = query.filter(
                DBCitation.reporter.in_(reporter_variations)
            ).all()

            for match in fuzzy_matches:
                # Score based on similarity
                score = 0.85  # Good confidence for volume+fuzzy_reporter+page
                if score > best_score:
                    best_match = match
                    best_score = score

    return best_match, best_score


def try_volume_page_match(
    session, volume: str, page: str
) -> tuple[object | None, float]:
    """
    Try to find a match using just volume and page.

    Args:
        session: Database session
        volume: Citation volume
        page: Citation page

    Returns:
        Tuple of (match, score) if found, (None, 0.0) otherwise
    """
    from sqlalchemy import and_

    best_match = None
    best_score = 0.0

    if volume and page:
        vol_page_matches = (
            session.query(DBCitation)
            .filter(and_(DBCitation.volume == volume, DBCitation.page == page))
            .all()
        )

        for match in vol_page_matches:
            score = 0.7  # Lower confidence for just volume+page
            if score > best_score:
                best_match = match
                best_score = score

    return best_match, best_score


def try_simplified_reporter_match(
    session, extracted_components: list
) -> tuple[object | None, float]:
    """
    Try to find a match using simplified reporter (first part only).

    Args:
        session: Database session
        extracted_components: List of dictionaries containing extracted components

    Returns:
        Tuple of (match, score) if found, (None, 0.0) otherwise
    """
    from sqlalchemy import and_

    best_match = None
    best_score = 0.0

    for component in extracted_components:
        reporter_raw = component["reporter"]
        if " " in reporter_raw:
            first_part = reporter_raw.split(" ")[0]
            if first_part.endswith("."):
                # This looks like a reporter abbreviation
                volume = component["volume"]
                page = component["page"]

                # Try to find matches with just the first part
                simplified_matches = (
                    session.query(DBCitation)
                    .filter(
                        and_(
                            DBCitation.volume == volume,
                            DBCitation.page == page,
                            DBCitation.reporter.like(f"{first_part}%"),
                        )
                    )
                    .all()
                )

                for match in simplified_matches:
                    score = 0.75  # Good confidence for simplified reporter match
                    if score > best_score:
                        best_match = match
                        best_score = score
                        logger.info(
                            f"Found match with simplified reporter: '{reporter_raw}' -> '{match.reporter}'"
                        )

    return best_match, best_score


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

        This improved version provides better error handling and more transparent defaults.

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

        # Track parsing issues for debugging
        parsing_issues = []

        try:
            # If already a CitationAnalysis, use directly
            if isinstance(citations, CitationAnalysis):
                return cls.from_citations([citations], cluster_id)

            # Handle list input - common case from LLM processing
            if isinstance(citations, list):
                # If it's a list of CitationAnalysis objects
                if all(isinstance(item, CitationAnalysis) for item in citations):
                    return cls.from_citations(citations, cluster_id)

                # If it's a list of dictionaries, try to convert each to CitationAnalysis
                citation_analyses = []
                for i, item in enumerate(citations):
                    if isinstance(item, dict):
                        try:
                            # Process each dictionary into a CitationAnalysis
                            citation_analyses.append(cls._process_citation_dict(item))
                        except Exception as e:
                            # Log specific field issues for better debugging
                            missing_fields = [
                                field
                                for field in [
                                    "date",
                                    "brief_summary",
                                    "majority_opinion_citations",
                                    "concurring_opinion_citations",
                                    "dissenting_citations",
                                ]
                                if field not in item
                            ]

                            issue = {
                                "item_index": i,
                                "error": str(e),
                                "missing_fields": missing_fields,
                                "item_keys": (
                                    list(item.keys())
                                    if isinstance(item, dict)
                                    else "not a dict"
                                ),
                            }
                            parsing_issues.append(issue)
                            logger.warning(f"Failed to process citation dict: {issue}")

                            # Create a minimal valid CitationAnalysis with explicit flags
                            try:
                                # Use more appropriate defaults
                                date_value = item.get("date")
                                if not date_value or date_value == "1500-01-01":
                                    # Try to extract year from cluster_id or use current year
                                    import datetime

                                    date_value = datetime.date.today().isoformat()

                                minimal_analysis = {
                                    "date": date_value,
                                    "brief_summary": item.get(
                                        "brief_summary",
                                        "[AUTO-GENERATED] Summary unavailable - partial data",
                                    ),
                                    "is_partial_data": True,  # Flag to indicate this is partial data
                                    "parsing_error": str(
                                        e
                                    ),  # Include the error message
                                    "majority_opinion_citations": [],
                                    "concurring_opinion_citations": [],
                                    "dissenting_citations": [],
                                }
                                citation_analyses.append(
                                    CitationAnalysis(**minimal_analysis)
                                )
                            except Exception as e2:
                                logger.warning(
                                    f"Failed to create minimal CitationAnalysis: {str(e2)}"
                                )

                if citation_analyses:
                    # Log if we had to use partial data
                    if parsing_issues:
                        logger.info(
                            f"Created CitationAnalysis with partial data for cluster {cluster_id}. Issues: {len(parsing_issues)}"
                        )
                    return cls.from_citations(citation_analyses, cluster_id)

                # If we couldn't process any items in the list, log detailed error and return None
                logger.error(
                    f"Could not process any items in citation list for cluster {cluster_id}. Issues: {parsing_issues}"
                )
                return None

            # If string, try to parse as JSON
            if isinstance(citations, str):
                try:
                    citation_data = json.loads(citations)
                except json.JSONDecodeError as e:
                    # Try to repair the JSON, but log the original error
                    logger.warning(
                        f"JSON parse error for cluster {cluster_id}: {str(e)}"
                    )
                    parsing_issues.append(
                        {"error_type": "json_decode", "error": str(e)}
                    )

                    try:
                        # Log the repair attempt
                        logger.info(
                            f"Attempting to repair JSON for cluster {cluster_id}"
                        )
                        repaired = repair_json(citations)
                        citation_data = json.loads(repaired)
                        logger.info(
                            f"Successfully repaired JSON for cluster {cluster_id}"
                        )
                    except Exception as repair_error:
                        logger.error(
                            f"JSON repair failed for cluster {cluster_id}: {str(repair_error)}"
                        )
                        # Instead of creating a default, return None to indicate failure
                        return None
            else:
                citation_data = citations

            # Handle dictionary input
            if isinstance(citation_data, dict):
                try:
                    citation_analysis = cls._process_citation_dict(citation_data)
                    return cls.from_citations([citation_analysis], cluster_id)
                except Exception as e:
                    logger.error(
                        f"Failed to process citation dictionary for cluster {cluster_id}: {str(e)}"
                    )
                    parsing_issues.append(
                        {"error_type": "dict_processing", "error": str(e)}
                    )

                    # Check if we have enough data for a partial result
                    if "date" in citation_data or "brief_summary" in citation_data:
                        try:
                            # Use more appropriate defaults
                            date_value = citation_data.get("date")
                            if not date_value or date_value == "1500-01-01":
                                import datetime

                                date_value = datetime.date.today().isoformat()

                            minimal_analysis = CitationAnalysis(
                                date=date_value,
                                brief_summary=citation_data.get(
                                    "brief_summary",
                                    "[AUTO-GENERATED] Partial data available",
                                ),
                                majority_opinion_citations=[],
                                concurring_opinion_citations=[],
                                dissenting_citations=[],
                            )
                            return cls.from_citations([minimal_analysis], cluster_id)
                        except Exception as e2:
                            logger.error(
                                f"Failed to create minimal CitationAnalysis for cluster {cluster_id}: {str(e2)}"
                            )
                    # If we don't have enough data, return None
                    return None

            # If we get here, the data format is invalid
            logger.error(
                f"Invalid citation data format for cluster {cluster_id}: {type(citation_data)}"
            )
            return None

        except Exception as e:
            logger.error(
                f"Failed to create CitationAnalysis for cluster {cluster_id}: {e}"
            )
            # Return None instead of creating a default with misleading data
            return None

    @classmethod
    def _process_citation_dict(cls, citation_data: Dict) -> CitationAnalysis:
        """
        Process a dictionary into a CitationAnalysis object.

        This improved version handles missing fields better and uses more appropriate defaults.

        Args:
            citation_data: Dictionary containing citation data

        Returns:
            CitationAnalysis object
        """
        import datetime

        # Track modifications made to the data
        modifications = []

        # Check required top-level fields but provide better defaults if missing
        if "date" not in citation_data or not citation_data["date"]:
            # Use current date instead of impossible date
            citation_data["date"] = datetime.date.today().isoformat()
            modifications.append("added_current_date")
            logger.info("Missing date field, using current date")
        elif citation_data["date"] == "1500-01-01":
            # Replace impossible date with current date
            citation_data["date"] = datetime.date.today().isoformat()
            modifications.append("replaced_impossible_date")
            logger.info("Found impossible date (1500-01-01), using current date")

        # Validate date format
        try:
            datetime.date.fromisoformat(citation_data["date"])
        except (ValueError, TypeError):
            # If date is invalid, use current date
            citation_data["date"] = datetime.date.today().isoformat()
            modifications.append("fixed_invalid_date_format")
            logger.info(
                f"Invalid date format: {citation_data.get('date')}, using current date"
            )

        if "brief_summary" not in citation_data or not citation_data["brief_summary"]:
            citation_data["brief_summary"] = "[AUTO-GENERATED] No summary available"
            modifications.append("added_missing_summary")
            logger.info("Missing brief_summary field, using placeholder")
        elif citation_data["brief_summary"].startswith("[DEFAULT]"):
            # Replace default marker with auto-generated marker
            citation_data["brief_summary"] = citation_data["brief_summary"].replace(
                "[DEFAULT]", "[AUTO-GENERATED]"
            )
            modifications.append("replaced_default_marker")

        # Add metadata about modifications if any were made
        if modifications:
            if "metadata" not in citation_data:
                citation_data["metadata"] = {}
            citation_data["metadata"]["modifications"] = modifications
            citation_data["metadata"][
                "modified_at"
            ] = datetime.datetime.now().isoformat()

        # Filter and clean citation lists
        def filter_citations(citations_list):
            if not isinstance(citations_list, list):
                logger.warning(
                    f"Expected list for citations but got {type(citations_list)}"
                )
                return []

            valid_citations = []
            invalid_citations = []

            for i, c in enumerate(citations_list):
                if not isinstance(c, dict):
                    invalid_citations.append(
                        {"index": i, "reason": f"Not a dictionary: {type(c)}"}
                    )
                    continue

                # Define required citation fields
                required_citation_fields = [
                    "page_number",
                    "citation_text",
                    "reasoning",
                    "type",
                    "treatment",
                    "relevance",
                ]

                # Track missing fields
                missing_fields = []

                # Add default values for missing fields
                for field in required_citation_fields:
                    if field not in c:
                        missing_fields.append(field)
                        if field == "page_number":
                            c[field] = None  # Use None instead of impossible value
                        elif field == "citation_text":
                            # Skip citations without citation text
                            invalid_citations.append(
                                {"index": i, "reason": "Missing citation_text"}
                            )
                            continue
                        elif field == "reasoning":
                            c[field] = "[AUTO-GENERATED] No reasoning provided"
                        elif field == "type":
                            c[field] = "other"
                        elif field == "treatment":
                            c[field] = "NEUTRAL"
                        elif field == "relevance":
                            c[field] = None  # Use None instead of impossible value

                # Convert string numeric fields to integers if needed
                if isinstance(c.get("page_number"), str):
                    try:
                        c["page_number"] = int(c["page_number"])
                    except ValueError:
                        c["page_number"] = None
                        missing_fields.append("invalid_page_number")

                if isinstance(c.get("relevance"), str):
                    try:
                        c["relevance"] = int(c["relevance"])
                    except ValueError:
                        c["relevance"] = None
                        missing_fields.append("invalid_relevance")

                # Add metadata about missing fields
                if missing_fields:
                    if "metadata" not in c:
                        c["metadata"] = {}
                    c["metadata"]["missing_fields"] = missing_fields
                    c["metadata"]["is_partial"] = True

                try:
                    valid_citations.append(Citation(**c))
                except Exception as e:
                    logger.warning(f"Failed to create Citation object: {str(e)}")
                    invalid_citations.append({"index": i, "reason": str(e), "data": c})

                    # Try to create a minimal valid Citation object
                    try:
                        minimal_citation = {
                            "page_number": c.get("page_number"),
                            "citation_text": c.get(
                                "citation_text", "[AUTO-GENERATED] Unknown citation"
                            ),
                            "reasoning": c.get(
                                "reasoning", "[AUTO-GENERATED] No reasoning provided"
                            ),
                            "type": c.get("type", "other"),
                            "treatment": c.get("treatment", "NEUTRAL"),
                            "relevance": c.get("relevance"),
                            "metadata": {
                                "is_minimal": True,
                                "original_error": str(e),
                                "missing_fields": missing_fields,
                            },
                        }
                        valid_citations.append(Citation(**minimal_citation))
                    except Exception as e2:
                        logger.warning(
                            f"Failed to create minimal Citation object: {str(e2)}"
                        )

            # Log statistics about citation processing
            if invalid_citations:
                logger.info(
                    f"Processed {len(valid_citations)} valid and {len(invalid_citations)} invalid citations"
                )

            return valid_citations

        # Add metadata about citation processing
        citation_data["metadata"] = citation_data.get("metadata", {})
        citation_data["metadata"][
            "processing_time"
        ] = datetime.datetime.now().isoformat()

        # Create CitationAnalysis object with filtered citations
        return CitationAnalysis(
            date=citation_data["date"],
            brief_summary=citation_data["brief_summary"],
            majority_opinion_citations=filter_citations(
                citation_data.get("majority_opinion_citations", [])
            ),
            concurring_opinion_citations=filter_citations(
                citation_data.get("concurring_opinion_citations", [])
            ),
            dissenting_citations=filter_citations(
                citation_data.get("dissenting_citations", [])
            ),
            metadata=citation_data.get("metadata", {}),
        )

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
