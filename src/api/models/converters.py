"""
Converters for API models.

This module provides essential converters for transforming between different model representations
when direct model inheritance or the from_* pattern isn't suitable.

# Converter Usage Guidelines

1. PREFER DIRECT MODEL USAGE: Whenever possible, import and use models directly from their
   source-of-truth location (src.llm_extraction.models, src.neo4j.models, src.postgres.models).

2. PREFER CLASS METHODS: For model conversion, prefer using class methods like `from_neo4j()` 
   defined on the target model class.

3. USE CONVERTERS SPARINGLY: Only use these converter functions when:
   - Working with legacy code that can't be easily refactored
   - Dealing with complex transformations that don't fit the class method pattern
   - Converting between incompatible model systems (e.g., SQLAlchemy to Pydantic)

4. DOCUMENT CLEARLY: Always document why a converter is needed instead of direct model usage.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import models from source of truth
from src.neo4j.models import Opinion as Neo4jOpinion
from src.postgres.models import OpinionClusterExtraction, CitationExtraction
from src.llm_extraction.models import OpinionSection, CitationType, CitationTreatment

logger = logging.getLogger(__name__)


def neo4j_opinion_to_core(opinion: Neo4jOpinion) -> Dict[str, Any]:
    """
    Convert a Neo4j Opinion node to a core dictionary representation.

    This utility function is used when direct model instantiation isn't possible,
    such as when working with legacy code or when the target model has validation
    that would reject a direct conversion.

    Args:
        opinion: Neo4j Opinion node

    Returns:
        Dictionary with core opinion data
    """
    if not opinion:
        logger.warning("Attempted to convert None to opinion dict")
        return {}

    try:
        return {
            "cluster_id": opinion.cluster_id,
            "case_name": opinion.case_name,
            "date_filed": (
                opinion.date_filed.isoformat() if opinion.date_filed else None
            ),
            "docket_id": opinion.docket_id,
            "docket_number": opinion.docket_number,
            "court_id": opinion.court_id,
            "court_name": opinion.court_name,
            "opinion_type": opinion.opinion_type,
            "scdb_votes_majority": opinion.scdb_votes_majority,
            "scdb_votes_minority": opinion.scdb_votes_minority,
            "ai_summary": opinion.ai_summary,
        }
    except Exception as e:
        logger.error(f"Error converting Neo4j opinion to dict: {str(e)}")
        return {"error": str(e)}


def postgres_citation_to_dict(citation: CitationExtraction) -> Dict[str, Any]:
    """
    Convert a PostgreSQL CitationExtraction to a dictionary representation.

    This is needed because SQLAlchemy models don't directly support Pydantic's
    model_dump() method, and we need to handle date/time serialization.

    Args:
        citation: SQLAlchemy CitationExtraction model

    Returns:
        Dictionary with citation data
    """
    if not citation:
        logger.warning("Attempted to convert None to citation dict")
        return {}

    try:
        return {
            "id": citation.id,
            "opinion_cluster_extraction_id": citation.opinion_cluster_extraction_id,
            "section": citation.section,
            "citation_type": citation.citation_type,
            "citation_text": citation.citation_text,
            "page_number": citation.page_number,
            "treatment": citation.treatment,
            "relevance": citation.relevance,
            "reasoning": citation.reasoning,
            "resolved_opinion_cluster": citation.resolved_opinion_cluster,
            "resolved_text": citation.resolved_text,
            "created_at": (
                citation.created_at.isoformat() if citation.created_at else None
            ),
            "updated_at": (
                citation.updated_at.isoformat() if citation.updated_at else None
            ),
        }
    except Exception as e:
        logger.error(f"Error converting PostgreSQL citation to dict: {str(e)}")
        return {"error": str(e)}


# Note: Many previous converters have been removed because models now directly
# use or extend the source-of-truth models and have their own from_* conversion methods.
