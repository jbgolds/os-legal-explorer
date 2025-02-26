from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from datetime import date
from neo4j import GraphDatabase

from ..database import get_neo4j
from ..models.stats import (
    NetworkStats, 
    CourtStats, 
    TimelineStats,
    TopCitedOpinions
)
from ..services import stats_service

router = APIRouter(
    prefix="/api/stats",
    tags=["stats"],
    responses={404: {"description": "Not found"}},
)

@router.get("/network", response_model=NetworkStats)
async def get_network_stats(neo4j_session = Depends(get_neo4j)):
    """
    Get overall statistics about the citation network.
    
    Returns:
        Overall network statistics
    """
    return stats_service.get_network_stats(neo4j_session)

@router.get("/courts", response_model=List[CourtStats])
async def get_court_stats(
    limit: int = Query(10, ge=1, le=50),
    neo4j_session = Depends(get_neo4j)
):
    """
    Get citation statistics by court.
    
    Args:
        limit: Maximum number of courts to return
        
    Returns:
        List of court statistics
    """
    return stats_service.get_court_stats(neo4j_session, limit=limit)

@router.get("/timeline", response_model=TimelineStats)
async def get_timeline_stats(
    court_id: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    neo4j_session = Depends(get_neo4j)
):
    """
    Get citation statistics over time.
    
    Args:
        court_id: Filter by court ID
        start_year: Start year for timeline
        end_year: End year for timeline
        
    Returns:
        Timeline statistics
    """
    return stats_service.get_timeline_stats(
        neo4j_session,
        court_id=court_id,
        start_year=start_year,
        end_year=end_year
    )

@router.get("/top-cited", response_model=TopCitedOpinions)
async def get_top_cited_opinions(
    court_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(20, ge=1, le=100),
    neo4j_session = Depends(get_neo4j)
):
    """
    Get the most cited opinions.
    
    Args:
        court_id: Filter by court ID
        start_date: Filter by date range (start)
        end_date: Filter by date range (end)
        limit: Maximum number of opinions to return
        
    Returns:
        List of top cited opinions
    """
    return stats_service.get_top_cited_opinions(
        neo4j_session,
        court_id=court_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )

@router.get("/citation-distribution")
async def get_citation_distribution(
    bins: int = Query(10, ge=5, le=50),
    neo4j_session = Depends(get_neo4j)
):
    """
    Get the distribution of citation counts.
    
    Args:
        bins: Number of bins for the distribution
        
    Returns:
        Citation count distribution
    """
    return stats_service.get_citation_distribution(
        neo4j_session,
        bins=bins
    )
