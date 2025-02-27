'''
Converters for API models.

This module provides converters for transforming between different model representations:
- Neo4j models (neomodel)
- PostgreSQL models (SQLAlchemy)
- API models (Pydantic)

This has been refactored to focus only on essential conversions that aren't handled
by direct model usage or inheritance.
'''

from datetime import datetime

# Import models from source of truth
from src.neo4j.models import Opinion as Neo4jOpinion
from src.postgres.models import OpinionClusterExtraction, CitationExtraction
from src.llm_extraction.models import OpinionSection, CitationType, CitationTreatment

def neo4j_opinion_to_core(opinion: Neo4jOpinion) -> dict:
    """
    Convert a Neo4j Opinion node to a core dictionary representation.
    
    This is a utility function for cases where direct model instantiation
    isn't possible or practical.
    
    Args:
        opinion: Neo4j Opinion node
        
    Returns:
        Dictionary with core opinion data
    """
    return {
        "cluster_id": opinion.cluster_id,
        "case_name": opinion.case_name,
        "date_filed": opinion.date_filed.isoformat() if opinion.date_filed else None,
        "docket_id": opinion.docket_id,
        "docket_number": opinion.docket_number,
        "court_id": opinion.court_id,
        "court_name": opinion.court_name,
        "opinion_type": opinion.opinion_type,
        "scdb_votes_majority": opinion.scdb_votes_majority,
        "scdb_votes_minority": opinion.scdb_votes_minority,
        "ai_summary": opinion.ai_summary,
    }

def postgres_citation_to_dict(citation: CitationExtraction) -> dict:
    """
    Convert a PostgreSQL CitationExtraction to a dictionary representation.
    
    Args:
        citation: SQLAlchemy CitationExtraction model
        
    Returns:
        Dictionary with citation data
    """
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
        "created_at": citation.created_at.isoformat() if citation.created_at else None,
        "updated_at": citation.updated_at.isoformat() if citation.updated_at else None,
    }

# Note: Many previous converters are no longer needed because models now directly
# use or extend the source-of-truth models and have their own from_* conversion methods. 