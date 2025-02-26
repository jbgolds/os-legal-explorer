from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from neo4j import GraphDatabase

from ..database import get_neo4j
from ..models.citations import (
    CitationNetwork, 
    CitationDetail, 
    CitationStats
)
from ..services import citation_service

router = APIRouter(
    prefix="/api/citations",
    tags=["citations"],
    responses={404: {"description": "Not found"}},
)

@router.get("/network", response_model=CitationNetwork)
async def get_citation_network(
    cluster_id: Optional[int] = None,
    court_id: Optional[str] = None,
    depth: int = Query(1, ge=1, le=3),
    limit: int = Query(100, ge=1, le=500),
    neo4j_session = Depends(get_neo4j)
):
    """
    Get a citation network centered around a specific opinion or court.
    
    Args:
        cluster_id: Center the network on this opinion cluster ID
        court_id: Filter by court ID
        depth: Network depth (1-3)
        limit: Maximum number of nodes to return
        
    Returns:
        Citation network with nodes and edges
    """
    if not cluster_id and not court_id:
        raise HTTPException(
            status_code=400, 
            detail="Either cluster_id or court_id must be provided"
        )
    
    return citation_service.get_citation_network(
        neo4j_session,
        cluster_id=cluster_id,
        court_id=court_id,
        depth=depth,
        limit=limit
    )

@router.get("/{citing_id}/{cited_id}", response_model=CitationDetail)
async def get_citation_detail(
    citing_id: int,
    cited_id: int,
    neo4j_session = Depends(get_neo4j)
):
    """
    Get detailed information about a specific citation relationship.
    
    Args:
        citing_id: The citing opinion cluster ID
        cited_id: The cited opinion cluster ID
        
    Returns:
        Detailed citation information
    """
    citation = citation_service.get_citation_detail(
        neo4j_session, 
        citing_id, 
        cited_id
    )
    if citation is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Citation relationship not found between {citing_id} and {cited_id}"
        )
    return citation

@router.get("/stats", response_model=CitationStats)
async def get_citation_stats(
    court_id: Optional[str] = None,
    year: Optional[int] = None,
    neo4j_session = Depends(get_neo4j)
):
    """
    Get citation statistics.
    
    Args:
        court_id: Filter by court ID
        year: Filter by year
        
    Returns:
        Citation statistics
    """
    return citation_service.get_citation_stats(
        neo4j_session,
        court_id=court_id,
        year=year
    )

@router.get("/influential", response_model=List[CitationDetail])
async def get_influential_citations(
    court_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    neo4j_session = Depends(get_neo4j)
):
    """
    Get the most influential citations based on citation count.
    
    Args:
        court_id: Filter by court ID
        limit: Maximum number of results to return
        
    Returns:
        List of influential citations
    """
    return citation_service.get_influential_citations(
        neo4j_session,
        court_id=court_id,
        limit=limit
    )
