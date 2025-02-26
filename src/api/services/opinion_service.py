from typing import List, Optional
from datetime import date
import logging

from ..models.opinions import OpinionResponse, OpinionDetail, OpinionBase
from src.neo4j.models import Opinion as Neo4jOpinion

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
    # Build the query based on filters
    query = {}
    
    if court_id:
        query["court_id"] = court_id
    
    if start_date:
        query["date_filed__gte"] = start_date
    
    if end_date:
        query["date_filed__lte"] = end_date
    
    try:
        # Get all opinions matching the query
        if query:
            opinions = Neo4jOpinion.nodes.filter(**query).order_by("-date_filed")
        else:
            opinions = Neo4jOpinion.nodes.order_by("-date_filed")
        
        # Apply pagination
        paginated_opinions = opinions[offset:offset + limit]
        
        # Convert to response models
        return [OpinionResponse.from_neo4j(opinion) for opinion in paginated_opinions]
    
    except Exception as e:
        logger.error(f"Error getting opinions: {str(e)}")
        return []

def get_opinion_by_cluster_id(db, cluster_id: int) -> Optional[OpinionDetail]:
    """
    Get detailed information about a specific opinion.
    
    Args:
        db: Database session (unused, kept for API compatibility)
        cluster_id: The opinion cluster ID
        
    Returns:
        Detailed opinion information or None if not found
    """
    try:
        # Get the opinion by cluster_id
        opinion = Neo4jOpinion.nodes.get(cluster_id=cluster_id)
        
        # Convert to response model
        return OpinionDetail.from_neo4j(opinion)
    
    except Neo4jOpinion.DoesNotExist:
        logger.warning(f"Opinion with cluster_id {cluster_id} not found")
        return None
    
    except Exception as e:
        logger.error(f"Error getting opinion {cluster_id}: {str(e)}")
        return None

def get_opinion_text(db, cluster_id: int) -> Optional[str]:
    """
    Get the full text of an opinion.
    
    Args:
        db: Database session (unused, kept for API compatibility)
        cluster_id: The opinion cluster ID
        
    Returns:
        Full text of the opinion or None if not found
    """
    try:
        # Get the opinion by cluster_id
        opinion = Neo4jOpinion.nodes.get(cluster_id=cluster_id)
        
        # Return the text if available
        # In a real implementation, this might be stored in a separate field or related node
        return opinion.metadata.get("text", "Opinion text not available.")
    
    except Neo4jOpinion.DoesNotExist:
        logger.warning(f"Opinion with cluster_id {cluster_id} not found")
        return None
    
    except Exception as e:
        logger.error(f"Error getting opinion text for {cluster_id}: {str(e)}")
        return None

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
    try:
        # Get the opinion by cluster_id
        opinion = Neo4jOpinion.nodes.get(cluster_id=cluster_id)
        
        # Get related opinions based on direction
        if direction == "outgoing":
            # Get opinions cited by this opinion
            related_opinions = opinion.cites.all()[:limit]
        else:
            # Get opinions citing this opinion
            related_opinions = opinion.cited_by.all()[:limit]
        
        # Convert to response models
        return [OpinionResponse.from_neo4j(related) for related in related_opinions]
    
    except Neo4jOpinion.DoesNotExist:
        logger.warning(f"Opinion with cluster_id {cluster_id} not found")
        return []
    
    except Exception as e:
        logger.error(f"Error getting citations for opinion {cluster_id}: {str(e)}")
        return []
