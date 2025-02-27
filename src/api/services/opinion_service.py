from typing import List, Optional
from datetime import date
import logging

from ..models.opinions import OpinionResponse, OpinionDetail, OpinionBase
from src.neo4j.models import Opinion as Neo4jOpinion
from .db_utils import get_opinion_by_id, get_filtered_opinions, get_postgres_entity_by_id

logger = logging.getLogger(__name__)

def get_opinions(
    db,
    court_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 20,
    offset: int = 0
) -> List[OpinionResponse]:
    """
    Get a list of opinions with optional filtering.
    
    Args:
        db: Database session (unused, kept for API compatibility)
        court_id: Filter by court ID
        start_date: Filter by date range (start)
        end_date: Filter by date range (end)
        limit: Maximum number of results to return
        offset: Number of results to skip
        
    Returns:
        List of opinion summaries
    """
    # Use the db_utils function for filtered queries
    opinions = get_filtered_opinions(
        court_id=court_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    
    # Convert to response models
    return [OpinionResponse.from_neo4j(opinion) for opinion in opinions]

def get_opinion_by_cluster_id(db, cluster_id: int) -> Optional[OpinionDetail]:
    """
    Get detailed information about a specific opinion.
    
    Args:
        db: Database session (unused, kept for API compatibility)
        cluster_id: The opinion cluster ID
        
    Returns:
        Detailed opinion information or None if not found
    """
    # Use the db_utils function to get entity by ID
    opinion = get_opinion_by_id(cluster_id)
    
    # Convert to response model if found
    if opinion:
        return OpinionDetail.from_neo4j(opinion)
    return None

def get_opinion_text(db, cluster_id: int) -> Optional[str]:
    """
    Get the full text of an opinion from PostgreSQL.
    
    Args:
        db: PostgreSQL database session
        cluster_id: The opinion cluster ID
        
    Returns:
        Full text of the opinion or None if not found
    """
    # Import models here to avoid circular imports
    from src.postgres.models import OpinionClusterExtraction, OpinionText
    
    # Get the opinion cluster using the db_utils function
    opinion_cluster = get_postgres_entity_by_id(
        db, 
        OpinionClusterExtraction, 
        "cluster_id", 
        cluster_id
    )
    
    if not opinion_cluster or not opinion_cluster.opinion_text:
        logger.warning(f"Opinion text for cluster_id {cluster_id} not found in PostgreSQL")
        return None
        
    return opinion_cluster.opinion_text.text

def get_opinion_citations(
    db,
    cluster_id: int,
    direction: str = "outgoing",
    limit: int = 20
) -> List[OpinionResponse]:
    """
    Get opinions cited by or citing a specific opinion.
    
    Args:
        db: Database session (unused, kept for API compatibility)
        cluster_id: The opinion cluster ID
        direction: "outgoing" for opinions cited by this opinion, 
                  "incoming" for opinions citing this opinion
        limit: Maximum number of results to return
        
    Returns:
        List of related opinions
    """
    # Use db_utils to get the opinion by ID
    opinion = get_opinion_by_id(cluster_id)
    
    if not opinion:
        return []
    
    try:
        # Get related opinions based on direction
        if direction == "outgoing":
            # Get opinions cited by this opinion
            related_opinions = opinion.cites.all()[:limit]
        else:
            # Get opinions citing this opinion
            related_opinions = opinion.cited_by.all()[:limit]
        
        # Convert to response models
        return [OpinionResponse.from_neo4j(related) for related in related_opinions]
    
    except Exception as e:
        logger.error(f"Error getting citations for opinion {cluster_id}: {str(e)}")
        return []
