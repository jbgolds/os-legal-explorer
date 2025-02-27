"""
Simple database utilities for service layer.

This module provides common database access patterns and error handling
to reduce duplication across service modules.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import date

from src.neo4j.models import Opinion as Neo4jOpinion

logger = logging.getLogger(__name__)

# --- Neo4j Utilities ---

def get_opinion_by_id(cluster_id: int) -> Optional[Neo4jOpinion]:
    """
    Get a Neo4j Opinion by cluster_id.
    
    Args:
        cluster_id: Opinion cluster ID
        
    Returns:
        Opinion object or None if not found
    """
    try:
        return Neo4jOpinion.nodes.get(cluster_id=cluster_id)
    except Neo4jOpinion.DoesNotExist:
        logger.warning(f"Opinion with cluster_id {cluster_id} not found")
        return None
    except Exception as e:
        logger.error(f"Error getting opinion {cluster_id}: {str(e)}")
        return None

def get_filtered_opinions(
    court_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    order_by: str = "-date_filed",
    limit: int = 20,
    offset: int = 0,
    **additional_filters
) -> List[Neo4jOpinion]:
    """
    Get filtered opinions with pagination.
    
    Args:
        court_id: Filter by court ID
        start_date: Filter by start date
        end_date: Filter by end date
        order_by: Field to order by
        limit: Max number of results
        offset: Number of results to skip
        additional_filters: Any additional filters
        
    Returns:
        List of opinions
    """
    try:
        # Build query filters
        filters = {}
        
        if court_id:
            filters["court_id"] = court_id
        
        if start_date:
            filters["date_filed__gte"] = start_date
        
        if end_date:
            filters["date_filed__lte"] = end_date
            
        # Add any additional filters
        filters.update(additional_filters)
        
        # Get filtered and ordered opinions
        if filters:
            opinions = Neo4jOpinion.nodes.filter(**filters)
        else:
            opinions = Neo4jOpinion.nodes
            
        if order_by:
            opinions = opinions.order_by(order_by)
            
        # Apply pagination
        return opinions[offset:offset + limit]
    
    except Exception as e:
        logger.error(f"Error getting filtered opinions: {str(e)}")
        return []

# --- PostgreSQL Utilities ---

def get_postgres_entity_by_id(db_session, model_class, id_field: str, id_value: Any):
    """
    Get a PostgreSQL entity by ID.
    
    Args:
        db_session: SQLAlchemy session
        model_class: Model class
        id_field: ID field name
        id_value: ID value
        
    Returns:
        Entity or None if not found
    """
    try:
        return db_session.query(model_class).filter(
            getattr(model_class, id_field) == id_value
        ).first()
    except Exception as e:
        logger.error(f"Error getting {model_class.__name__} with {id_field}={id_value}: {str(e)}")
        return None 