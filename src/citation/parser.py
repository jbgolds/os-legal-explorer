"""
Citation parsing and resolution utilities.

This module centralizes all citation parsing and resolution logic to avoid duplication.
It provides functions for:
1. Parsing citations using eyecite
2. Extracting citation components using regex patterns
3. Resolving citations to cluster IDs using database lookups
4. Fuzzy matching for citations that don't match exactly
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy import and_
from eyecite import get_citations
from eyecite.resolve import resolve_citations
from reporters_db import REPORTERS, VARIATIONS_ONLY
from src.postgres.database import get_db_session, Citation

logger = logging.getLogger(__name__)

# Create a global reporter lookup dictionary - computed once and reused
_REPORTER_LOOKUP = None


def get_reporter_lookup() -> Dict[str, str]:
    """
    Create a dictionary mapping reporter variations to their canonical forms.
    Uses reporters-db's VARIATIONS_ONLY and REPORTERS to build a comprehensive mapping.
    This function caches the result for efficiency.

    Returns:
        Dictionary mapping reporter variations to canonical names
    """
    global _REPORTER_LOOKUP

    # Return cached version if available
    if _REPORTER_LOOKUP is not None:
        return _REPORTER_LOOKUP

    lookup = {}

    # Build lookup dictionary directly from VARIATIONS_ONLY
    # This is the primary source of variation-to-canonical mappings
    for variation, canonical_list in VARIATIONS_ONLY.items():
        if canonical_list:  # Make sure we have a canonical form
            canonical = canonical_list[0]
            lookup[variation.lower()] = canonical

            # Also add the canonical form mapping to itself
            lookup[canonical.lower()] = canonical

    # Add direct mappings for all reporter keys from REPORTERS
    for reporter_key in REPORTERS.keys():
        lookup[reporter_key.lower()] = reporter_key

    # Add reporter names from REPORTERS structure
    for reporter_key, reporter_data_list in REPORTERS.items():
        for reporter_data in reporter_data_list:
            name = reporter_data.get("name", "")
            if name:
                lookup[name.lower()] = reporter_key

    # Cache the result for future calls
    _REPORTER_LOOKUP = lookup
    return lookup


def normalize_reporter(
    reporter_raw: str, reporter_lookup: Optional[Dict[str, str]] = None
) -> str:
    """
    Normalize reporter abbreviation to canonical form using reporters-db's data.

    This implementation preserves the specific edition information (like 2d, 3d)
    which is critical for accurate legal citation.

    Args:
        reporter_raw: Raw reporter string
        reporter_lookup: Optional dictionary mapping variations to canonical names
                         (if None, the global lookup will be used)

    Returns:
        Normalized reporter string
    """
    if not reporter_raw:
        return ""

    reporter_lower = reporter_raw.lower().strip()

    # Use provided lookup or get the global one
    if reporter_lookup is None:
        reporter_lookup = get_reporter_lookup()

    # Direct lookup - most efficient approach
    if reporter_lower in reporter_lookup:
        return reporter_lookup[reporter_lower]

    # If we couldn't find it in the lookup dictionary, return the original
    return reporter_raw


# Legacy function for API compatibility - now just returns the result of get_reporter_lookup
def create_reporter_lookup(
    reporter_mappings: Optional[Dict[str, List[str]]] = None
) -> Dict[str, str]:
    """
    Legacy function that returns a reporter lookup dictionary.
    Now just a wrapper around get_reporter_lookup for API compatibility.

    Args:
        reporter_mappings: Optional dictionary mapping canonical names to variations
                          (ignored in current implementation)

    Returns:
        Dictionary mapping variations to canonical names
    """
    # If custom mappings provided, use those instead
    if reporter_mappings is not None:
        lookup = {}
        for canonical, variations in reporter_mappings.items():
            for variation in variations:
                lookup[variation.lower()] = canonical
        return lookup

    # Otherwise use the global lookup
    return get_reporter_lookup()


# Legacy function for API compatibility - builds reporter mappings from REPORTERS
def get_reporter_mappings() -> Dict[str, List[str]]:
    """
    Legacy function that builds reporter mappings from REPORTERS.
    Kept for API compatibility.

    Returns:
        Dictionary mapping canonical reporter names to lists of variations
    """
    reporter_mappings = {}
    for reporter_key, reporter_data_list in REPORTERS.items():
        if reporter_key not in reporter_mappings:
            reporter_mappings[reporter_key] = []

        for reporter_data in reporter_data_list:
            # Add the name as a variation
            name = reporter_data.get("name", "")
            if name and name not in reporter_mappings[reporter_key]:
                reporter_mappings[reporter_key].append(name)

            # Add all variations
            variations = reporter_data.get("variations", {})
            for variation in variations:
                if variation not in reporter_mappings[reporter_key]:
                    reporter_mappings[reporter_key].append(variation)

    return reporter_mappings


# Citation resolution functions


def find_cluster_id(citation_string: str) -> Optional[int]:
    """
    Find cluster_id for a given citation string using eyecite for parsing.

    Args:
        citation_string: Citation text to look up

    Returns:
        Optional[int]: Cluster ID if found, None otherwise
    """
    if not citation_string:
        raise ValueError("Citation string is required")

    try:
        # Extract citations using eyecite
        citations = get_citations(citation_string)
        if not citations:
            logger.debug(f"No citations found in: {citation_string}")
            return None

        # Resolve the citations
        resolved_citations = resolve_citations(citations)
        if not resolved_citations:
            logger.debug(f"Could not resolve citations in: {citation_string}")
            return None

        # Use the first resolved citation
        resolved = list(resolved_citations.keys())[0]
        resolved_citation = resolved.citation

        # Extract normalized components
        volume = str(resolved_citation.groups["volume"])
        reporter = resolved_citation.groups["reporter"]
        page = str(resolved_citation.groups["page"])

        if not volume or not reporter or not page:
            logger.warning(
                f"Invalid citation lookup: {citation_string}, missing volume, reporter, or page"
            )
            return None

        # Use database session with proper error handling

        with get_db_session() as session:
            # Query the database with resolved components
            result = (
                session.query(Citation)
                .filter(
                    and_(
                        Citation.volume == volume,
                        Citation.reporter == reporter,
                        Citation.page == page,
                    )
                )
                .first()
            )

            return result.cluster_id if result else None

    except Exception as e:
        logger.error(f"Error processing citation '{citation_string}': {str(e)}")
        return None


def clean_citation_text(text: str) -> str:
    """
    Clean citation text by removing extra spaces and normalizing punctuation.

    Args:
        text: Citation text to clean

    Returns:
        Cleaned citation text
    """
    if not text:
        return ""

    # Remove extra spaces
    cleaned = re.sub(r"\s+", " ", text.strip())

    # Normalize punctuation
    cleaned = re.sub(r"[\.,;:]\s*[\.,;:]", ".", cleaned)

    # Ensure consistent spacing around punctuation
    cleaned = re.sub(r"\s*([\.,;:])\s*", r"\1 ", cleaned)

    return cleaned


def extract_citation_components(citation_text: str) -> List[Dict[str, str]]:
    """
    Extract citation components (volume, reporter, page) from citation text.

    Args:
        citation_text: Citation text to parse

    Returns:
        List of dictionaries containing extracted components
    """
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


def extract_parallel_citations(citation_text: str) -> List[Dict[str, str]]:
    """
    Extract components from parallel citations.

    Args:
        citation_text: Citation text to parse

    Returns:
        List of dictionaries containing extracted components
    """
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


def find_cluster_id_fuzzy(
    citation_text: str, threshold: float = 0.8
) -> Tuple[Optional[int], float]:
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
    logger.info(f"Attempting fuzzy match for: {citation_text}")

    # Define reporter mappings
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


def find_best_match(
    extracted_components: List[Dict[str, str]],
    reporter_mappings: Dict[str, List[str]],
    reporter_lookup: Dict[str, str],
    threshold: float,
    citation_text: str,
) -> Tuple[Optional[int], float]:
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
    from src.postgres.database import get_db_session, Citation

    best_match = None
    best_score = 0.0

    with get_db_session() as session:
        # Try different matching strategies for each extracted component set
        for components in extracted_components:
            volume = components.get("volume", "")
            reporter_raw = components.get("reporter", "")
            page = components.get("page", "")

            if not volume or not reporter_raw or not page:
                continue

            # 1. Try exact match first
            match, score = try_exact_match(session, volume, reporter_raw, page)
            if match and score > best_score:
                best_match = match
                best_score = score
                logger.info(f"Found exact match: {match.cluster_id} with score {score}")
                if score >= 0.95:  # If we have a very good match, return immediately
                    return match.cluster_id, score

            # 2. Try fuzzy reporter matching
            match, score = try_fuzzy_reporter_match(
                session, volume, reporter_raw, page, reporter_mappings
            )
            if match and score > best_score:
                best_match = match
                best_score = score
                logger.info(
                    f"Found fuzzy reporter match: {match.cluster_id} with score {score}"
                )
                if score >= 0.9:  # If we have a good match, return immediately
                    return match.cluster_id, score

            # 3. Try volume and page matching only (for cases where reporter is misidentified)
            match, score = try_volume_page_match(session, volume, page)
            if match and score > best_score:
                best_match = match
                best_score = score
                logger.info(
                    f"Found volume-page match: {match.cluster_id} with score {score}"
                )

            # 4. Try simplified reporter matching
            match, score = try_simplified_reporter_match(session, extracted_components)
            if match and score > best_score:
                best_match = match
                best_score = score
                logger.info(
                    f"Found simplified reporter match: {match.cluster_id} with score {score}"
                )

    # Return the best match if it meets the threshold
    if best_match and best_score >= threshold:
        return best_match.cluster_id, best_score
    else:
        logger.warning(
            f"No match found for citation '{citation_text}' with score >= {threshold}"
        )
        return None, 0.0


def try_exact_match(
    session, volume: str, reporter: str, page: str
) -> Tuple[Optional[Any], float]:
    """
    Try to find an exact match for the citation components.

    Args:
        session: Database session
        volume: Citation volume
        reporter: Citation reporter
        page: Citation page

    Returns:
        Tuple of (match object, confidence score)
    """
    from src.postgres.database import Citation

    try:
        result = (
            session.query(Citation)
            .filter(
                and_(
                    Citation.volume == volume,
                    Citation.reporter == reporter,
                    Citation.page == page,
                )
            )
            .first()
        )
        return (result, 1.0) if result else (None, 0.0)
    except Exception as e:
        logger.error(f"Error in exact match: {str(e)}")
        return None, 0.0


def try_fuzzy_reporter_match(
    session,
    volume: str,
    reporter: str,
    page: str,
    reporter_mappings: Dict[str, List[str]],
) -> Tuple[Optional[Any], float]:
    """
    Try to find a match using fuzzy reporter matching.

    Args:
        session: Database session
        volume: Citation volume
        reporter: Citation reporter
        page: Citation page
        reporter_mappings: Dictionary mapping canonical reporter names to variations

    Returns:
        Tuple of (match object, confidence score)
    """
    from src.postgres.database import Citation

    try:
        # Find all possible variations of the reporter
        possible_reporters = []
        reporter_lower = reporter.lower()

        for canonical, variations in reporter_mappings.items():
            for variation in variations:
                if (
                    variation.lower() in reporter_lower
                    or reporter_lower in variation.lower()
                ):
                    possible_reporters.append(canonical)
                    possible_reporters.extend(variations)

        # Remove duplicates
        possible_reporters = list(set(possible_reporters))

        if not possible_reporters:
            return None, 0.0

        # Query with all possible reporters
        result = (
            session.query(Citation)
            .filter(
                and_(
                    Citation.volume == volume,
                    Citation.reporter.in_(possible_reporters),
                    Citation.page == page,
                )
            )
            .first()
        )

        return (result, 0.9) if result else (None, 0.0)
    except Exception as e:
        logger.error(f"Error in fuzzy reporter match: {str(e)}")
        return None, 0.0


def try_volume_page_match(
    session, volume: str, page: str
) -> Tuple[Optional[Any], float]:
    """
    Try to find a match using only volume and page.

    Args:
        session: Database session
        volume: Citation volume
        page: Citation page

    Returns:
        Tuple of (match object, confidence score)
    """
    from src.postgres.database import Citation

    try:
        result = (
            session.query(Citation)
            .filter(
                and_(
                    Citation.volume == volume,
                    Citation.page == page,
                )
            )
            .first()
        )

        return (result, 0.7) if result else (None, 0.0)
    except Exception as e:
        logger.error(f"Error in volume-page match: {str(e)}")
        return None, 0.0


def try_simplified_reporter_match(
    session, extracted_components: List[Dict[str, str]]
) -> Tuple[Optional[Any], float]:
    """
    Try to find a match using simplified reporter strings.

    Args:
        session: Database session
        extracted_components: List of dictionaries containing extracted components

    Returns:
        Tuple of (match object, confidence score)
    """
    from src.postgres.database import Citation

    try:
        # Extract all possible volume-page combinations
        volume_page_pairs = []
        for comp in extracted_components:
            volume = comp.get("volume")
            page = comp.get("page")
            if volume and page:
                volume_page_pairs.append((volume, page))

        if not volume_page_pairs:
            return None, 0.0

        # Try each volume-page pair
        for volume, page in volume_page_pairs:
            result = (
                session.query(Citation)
                .filter(
                    and_(
                        Citation.volume == volume,
                        Citation.page == page,
                    )
                )
                .first()
            )

            if result:
                return result, 0.6

        return None, 0.0
    except Exception as e:
        logger.error(f"Error in simplified reporter match: {str(e)}")
        return None, 0.0
