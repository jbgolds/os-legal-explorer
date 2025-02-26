from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date

from ..database import get_db
from ..models.opinions import OpinionResponse, OpinionDetail
from ..services import opinion_service

router = APIRouter(
    prefix="/api/opinions",
    tags=["opinions"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=List[OpinionResponse])
async def get_opinions(
    court_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get a list of opinions with optional filtering.
    
    Args:
        court_id: Filter by court ID
        start_date: Filter by date range (start)
        end_date: Filter by date range (end)
        limit: Maximum number of results to return
        offset: Number of results to skip
        
    Returns:
        List of opinion summaries
    """
    return opinion_service.get_opinions(
        db, 
        court_id=court_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )

@router.get("/{cluster_id}", response_model=OpinionDetail)
async def get_opinion(cluster_id: int, db: Session = Depends(get_db)):
    """
    Get detailed information about a specific opinion.
    
    Args:
        cluster_id: The opinion cluster ID
        
    Returns:
        Detailed opinion information
    """
    opinion = opinion_service.get_opinion_by_cluster_id(db, cluster_id)
    if opinion is None:
        raise HTTPException(status_code=404, detail="Opinion not found")
    return opinion

@router.get("/{cluster_id}/text")
async def get_opinion_text(cluster_id: int, db: Session = Depends(get_db)):
    """
    Get the full text of an opinion.
    
    Args:
        cluster_id: The opinion cluster ID
        
    Returns:
        Full text of the opinion
    """
    text = opinion_service.get_opinion_text(db, cluster_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Opinion text not found")
    return {"cluster_id": cluster_id, "text": text}

@router.get("/{cluster_id}/citations", response_model=List[OpinionResponse])
async def get_opinion_citations(
    cluster_id: int, 
    direction: str = Query("outgoing", regex="^(outgoing|incoming)$"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get opinions cited by or citing a specific opinion.
    
    Args:
        cluster_id: The opinion cluster ID
        direction: "outgoing" for opinions cited by this opinion, 
                  "incoming" for opinions citing this opinion
        limit: Maximum number of results to return
        
    Returns:
        List of related opinions
    """
    return opinion_service.get_opinion_citations(
        db, 
        cluster_id, 
        direction=direction,
        limit=limit
    )
