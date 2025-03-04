import logging
from typing import Optional, List, Dict, Any, Union
from fastapi import APIRouter, HTTPException, Query, Request, Depends
import httpx
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
from fastapi.responses import HTMLResponse, JSONResponse
from ..shared import templates
from src.neo4j_db.models import Opinion

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/case",
    tags=["cases"],
    responses={404: {"description": "Not found"}},
)


# Models
class Citation(BaseModel):
    id: str
    name: str
    citation: Optional[str] = None
    treatment: Optional[str] = None  # positive, negative, neutral, etc.


class Case(BaseModel):
    cluster_id: str
    case_name: str
    court_name: str
    date_filed: datetime
    citation: Optional[str] = None
    html: Optional[str] = None
    plain_text: Optional[str] = None


class CaseDetail(BaseModel):
    id: str
    name: str
    court_name: str
    date_filed: Optional[datetime] = None
    docket_number: Optional[str] = None
    citation: Optional[str] = None
    judges: Optional[List[str]] = None
    opinion_text: Optional[str] = None
    citations: Optional[List[Citation]] = None


class CitationNode(BaseModel):
    id: str
    name: str
    court: str
    year: Optional[int] = None
    type: str = "case"  # case, statute, etc.


class CitationEdge(BaseModel):
    source: str
    target: str
    treatment: str  # positive, negative, neutral, etc.
    context: Optional[str] = None


class CitationNetwork(BaseModel):
    nodes: List[CitationNode]
    edges: List[CitationEdge]


# Add new model for case status response
class CaseStatus(BaseModel):
    exists: bool
    has_citations: bool
    citation_count: Optional[int] = None


# CourtListener API endpoint
COURTLISTENER_API_URL = "https://www.courtlistener.com/api/rest/v4"
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY", "")


# Case detail endpoint
@router.get("/{case_id}", response_model=CaseDetail)
async def get_case_details_api(case_id: str):
    """
    Get detailed information for a specific case.

    Parameters:
    - case_id: The ID of the case to retrieve
    """
    try:
        # TODO: Implement Neo4j query to fetch case details
        # For now, we'll use the CourtListener API as a temporary solution

        # Make request to CourtListener API
        async with httpx.AsyncClient() as client:
            # Set up headers with API key if available
            headers = {}
            if COURTLISTENER_API_KEY:
                headers["Authorization"] = f"Token {COURTLISTENER_API_KEY}"

            # Get opinion cluster details
            response = await client.get(
                f"{COURTLISTENER_API_URL}/clusters/{case_id}/",
                headers=headers,
                timeout=30.0,
            )

            # Check for errors
            response.raise_for_status()
            cluster_data = response.json()

            # Get the first opinion from the cluster
            opinion_id = None
            if cluster_data.get("sub_opinions"):
                opinion_id = cluster_data["sub_opinions"][0]

            opinion_text = None
            if opinion_id:
                # Get opinion details
                opinion_response = await client.get(
                    f"{COURTLISTENER_API_URL}/opinions/{opinion_id}/",
                    headers=headers,
                    timeout=30.0,
                )
                opinion_response.raise_for_status()
                opinion_data = opinion_response.json()
                opinion_text = opinion_data.get("plain_text", "")

            # Extract judges
            judges = []
            if cluster_data.get("panel"):
                judges = [judge.strip() for judge in cluster_data["panel"].split(",")]

            # Create case detail object
            case_detail = CaseDetail(
                id=case_id,
                name=cluster_data.get("case_name", ""),
                court_name=cluster_data.get("court", {}).get("full_name", ""),
                date_filed=cluster_data.get("date_filed"),
                docket_number=cluster_data.get("docket_number", ""),
                citation=cluster_data.get("citation_string", ""),
                judges=judges,
                opinion_text=opinion_text,
                citations=[],  # We'll populate this in a future implementation
            )

            return case_detail

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching case details: {e}")
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Case not found")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error from CourtListener API: {e.response.text}",
        )
    except httpx.RequestError as e:
        logger.error(f"Request error fetching case details: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error connecting to CourtListener API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching case details: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# Citation network endpoint
@router.get("/{case_id}/citations", response_model=CitationNetwork)
async def get_citation_network_api(
    case_id: str,
    depth: int = Query(1, ge=1, le=3),
):
    """
    Get the citation network for a specific case.

    Parameters:
    - case_id: The ID of the case to retrieve the citation network for
    - depth: The depth of the citation network (default: 1, max: 3)
    """
    try:
        # TODO: Implement Neo4j query to fetch citation network
        # For now, we'll return a placeholder network

        # Create a placeholder citation network
        nodes = [
            CitationNode(
                id=case_id,
                name="Current Case",
                court="Unknown Court",
                year=2023,
                type="case",
            )
        ]

        edges = []

        # Add some placeholder nodes and edges
        for i in range(1, 6):
            node_id = f"placeholder-{i}"
            nodes.append(
                CitationNode(
                    id=node_id,
                    name=f"Placeholder Case {i}",
                    court=f"Court {i}",
                    year=2020 - i,
                    type="case",
                )
            )

            # Add edge from current case to placeholder
            treatment = (
                "positive" if i % 3 == 0 else "neutral" if i % 3 == 1 else "negative"
            )
            edges.append(
                CitationEdge(
                    source=case_id,
                    target=node_id,
                    treatment=treatment,
                    context=f"Citation context {i}",
                )
            )

            # Add some connections between placeholder nodes
            if i > 1:
                edges.append(
                    CitationEdge(
                        source=node_id,
                        target=f"placeholder-{i-1}",
                        treatment="neutral",
                        context=f"Related citation {i}",
                    )
                )

        return CitationNetwork(nodes=nodes, edges=edges)

    except Exception as e:
        logger.error(f"Unexpected error fetching citation network: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/{cluster_id}/details", response_class=HTMLResponse)
async def get_case_details_component(request: Request, cluster_id: str):
    """Get case details and render the case detail component."""
    try:
        # TODO: Replace with actual database query
        case = Case(
            cluster_id=cluster_id,
            case_name="Sample Case Name",
            court_name="Sample Court",
            date_filed=datetime.now(),
            citation="123 F.3d 456",
            plain_text="Sample case text...",
        )
        return templates.TemplateResponse(
            "components/case_detail.html",
            {"request": request, "case": case},
        )
    except Exception as e:
        logger.error(f"Error getting case details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{cluster_id}/citations/component", response_class=HTMLResponse)
async def get_case_citations_component(request: Request, cluster_id: str):
    """Get citation network visualization component for a case."""
    try:
        # TODO: Replace with actual citation network query from Neo4j
        citations = [
            Citation(
                id="1", name="Case 1", citation="123 F.3d 456", treatment="positive"
            ),
            Citation(
                id="2", name="Case 2", citation="789 F.3d 012", treatment="negative"
            ),
        ]
        return templates.TemplateResponse(
            "components/citation_network.html",
            {"request": request, "case_id": cluster_id, "citations": citations},
        )
    except Exception as e:
        logger.error(f"Error getting citation network: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{cluster_id}/related", response_class=HTMLResponse)
async def get_related_cases(request: Request, cluster_id: str):
    """Get related cases component."""
    try:
        # TODO: Replace with actual related cases query
        related_cases = [
            Case(
                cluster_id="1",
                case_name="Related Case 1",
                court_name="Court 1",
                date_filed=datetime.now(),
            ),
            Case(
                cluster_id="2",
                case_name="Related Case 2",
                court_name="Court 2",
                date_filed=datetime.now(),
            ),
        ]
        return templates.TemplateResponse(
            "components/related_cases.html",
            {"request": request, "cases": related_cases},
        )
    except Exception as e:
        logger.error(f"Error getting related cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{case_id}/status")
async def check_case_status(case_id: str) -> CaseStatus:
    """
    Check if a case exists in Neo4j and has outgoing citations.

    Parameters:
    - case_id: The ID of the case to check

    Returns:
    - CaseStatus object with existence and citation information
    """
    try:
        # Try to find the opinion by primary_id (cluster_id)
        opinion = Opinion.nodes.first_or_none(primary_id=case_id)

        if not opinion:
            return CaseStatus(exists=False, has_citations=False)

        # Count outgoing citations
        citation_count = len(opinion.cites.all())

        return CaseStatus(
            exists=True, has_citations=citation_count > 0, citation_count=citation_count
        )

    except Exception as e:
        logger.error(f"Error checking case status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error checking case status: {str(e)}"
        )


@router.get("/{case_id}/citation-network", response_model=CitationNetwork)
async def get_citation_network_data(case_id: str) -> CitationNetwork:
    """
    Get the outgoing citation network for a case.

    Parameters:
    - case_id: The ID of the case to get citations for

    Returns:
    - CitationNetwork object containing nodes and edges of the citation network
    """
    try:
        # Find the source opinion
        opinion = Opinion.nodes.first_or_none(primary_id=case_id)
        if not opinion:
            raise HTTPException(status_code=404, detail="Case not found")

        # Get all outgoing citations
        cited_opinions = opinion.cites.all()

        # Create nodes list starting with the source node
        nodes = [
            CitationNode(
                id=case_id,
                name=opinion.case_name or "Unknown Case",
                court=opinion.court_name or "Unknown Court",
                year=opinion.date_filed.year if opinion.date_filed else None,
                type="case",
            )
        ]

        # Create edges list and add cited nodes
        edges = []
        for rel in opinion.cites:
            # Get the target node
            target_node = rel.end_node()

            # Add target node to nodes list
            nodes.append(
                CitationNode(
                    id=target_node.primary_id,
                    name=target_node.case_name or "Unknown Case",
                    court=target_node.court_name or "Unknown Court",
                    year=(
                        target_node.date_filed.year if target_node.date_filed else None
                    ),
                    type="case",
                )
            )

            # Add the edge with its metadata
            edges.append(
                CitationEdge(
                    source=case_id,
                    target=target_node.primary_id,
                    treatment=rel.treatment or "neutral",
                    context=rel.citation_text,
                )
            )

        return CitationNetwork(nodes=nodes, edges=edges)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting citation network: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting citation network: {str(e)}"
        )
