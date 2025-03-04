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
from src.neo4j_db.neomodel_loader import NeomodelLoader
from neo4j.exceptions import ServiceUnavailable, AuthError
from neo4j import GraphDatabase
from neomodel import db

# Load environment variables
load_dotenv()

loader = NeomodelLoader()
# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/opinion",
    tags=["opinions"],
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

    @classmethod
    def from_legal_doc(cls, doc) -> "CitationNode":
        """Create a CitationNode from any legal document type"""
        return cls(
            id=str(doc.primary_id),
            name=(
                str(doc.case_name)
                if hasattr(doc, "case_name")
                else str(doc.name) if hasattr(doc, "name") else "Unknown Document"
            ),
            court=(
                str(doc.court_name) if hasattr(doc, "court_name") else "Unknown Court"
            ),
            year=(
                int(doc.date_filed.strftime("%Y"))
                if hasattr(doc, "date_filed") and doc.date_filed
                else None
            ),
            type=doc.__class__.__name__.lower(),
        )


class CitationEdge(BaseModel):
    source: str
    target: str
    treatment: str  # positive, negative, neutral, etc.
    reasoning: str

    @classmethod
    def from_relationship(cls, source_id: str, relationship) -> "CitationEdge":
        """Create a CitationEdge from a Neo4j relationship"""
        return cls(
            source=str(source_id),
            target=str(relationship.end_node().primary_id),
            treatment=(
                str(relationship.treatment)
                if hasattr(relationship, "treatment")
                else "neutral"
            ),
            reasoning=(
                str(relationship.reasoning)
                if hasattr(relationship, "reasoning")
                else "No reasoning provided"
            ),
        )


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


async def get_case_from_courtlistener(case_id: str) -> Optional[CaseDetail]:
    """
    Fetch case details from CourtListener API
    """
    try:
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

            # Create case detail object
            case = CaseDetail(
                id=case_id,
                name=cluster_data.get("case_name", "Unknown Case"),
                court_name=cluster_data.get("court", {}).get(
                    "full_name", "Unknown Court"
                ),
                date_filed=(
                    datetime.fromisoformat(cluster_data.get("date_filed"))
                    if cluster_data.get("date_filed")
                    else None
                ),
                docket_number=cluster_data.get("docket_number"),
                citation=cluster_data.get("citation", {}).get("cite_string"),
                judges=cluster_data.get("panel", []),
                opinion_text=opinion_text,
                citations=[],  # TODO: Implement citation extraction
            )

            return case

    except httpx.HTTPError as e:
        logger.error(f"Error fetching case from CourtListener: {e}")
        return None


# Case detail endpoint
@router.get("/{opinion_id}", response_model=CaseDetail)
async def get_case_details_api(opinion_id: str):
    """
    Get detailed information for a specific case.
    First tries to get data from Neo4j, falls back to CourtListener API if not found.

    Parameters:
    - opinion_id: The ID of the case to retrieve
    """
    try:
        # Try to get case from Neo4j
        opinion = NeomodelLoader().get_case_from_neo4j(opinion_id)

        if opinion:
            # Convert Neo4j opinion to CaseDetail
            case = CaseDetail(
                id=opinion_id,
                name=str(opinion.case_name) if opinion.case_name else "Unknown Case",
                court_name=(
                    str(opinion.court_name) if opinion.court_name else "Unknown Court"
                ),
                date_filed=(
                    datetime.combine(opinion.date_filed, datetime.min.time())
                    if opinion.date_filed
                    else None
                ),
                docket_number=(
                    str(opinion.docket_number) if opinion.docket_number else None
                ),
                citation=(
                    str(opinion.citation_string) if opinion.citation_string else None
                ),
                judges=[],  # TODO: Add judges from Neo4j
                opinion_text=None,  # TODO: Get from PostgreSQL
                citations=[],  # TODO: Get from Neo4j relationships
            )
            return case

    except (ServiceUnavailable, AuthError) as e:
        logger.error(f"Neo4j database error: {e}")
        # Continue to CourtListener fallback

    # If not found in Neo4j or error occurred, try CourtListener
    case = await get_case_from_courtlistener(opinion_id)
    if case:
        return case

    # If not found in either source
    raise HTTPException(status_code=404, detail="Case not found")


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

        # Count outgoing citations in a separate transaction
        try:
            with db.transaction:
                citation_count = len(list(opinion.cites))

            return CaseStatus(
                exists=True,
                has_citations=citation_count > 0,
                citation_count=citation_count,
            )
        except Exception as e:
            logger.error(f"Error counting citations: {e}")
            return CaseStatus(exists=True, has_citations=False)

    except Exception as e:
        logger.error(f"Error checking case status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error checking case status: {str(e)}"
        )


@router.get("/{case_id}/citation-network", response_model=CitationNetwork)
async def get_citation_network_data(case_id: str) -> CitationNetwork:
    """
    Get the citation network for any legal document, including all connected nodes and relationships.

    Parameters:
    - case_id: The ID of the legal document to get citations for

    Returns:
    - CitationNetwork object containing nodes and edges of the citation network
    """
    try:
        # Get the source node
        source_doc = NeomodelLoader().get_case_from_neo4j(case_id)
        if not source_doc:
            logger.warning(f"No document found for id: {case_id}")
            return CitationNetwork(nodes=[], edges=[])

        nodes = []
        edges = []

        # Add the source node
        source_node = CitationNode.from_legal_doc(source_doc)
        nodes.append(source_node)

        try:
            # Get all outgoing citation relationships
            relationships = source_doc.cites.all()

            # Process each relationship
            for rel in relationships:
                # Add the cited document node
                cited_node = CitationNode.from_legal_doc(rel.end_node())
                nodes.append(cited_node)

                # Add the edge
                edge = CitationEdge.from_relationship(source_doc.primary_id, rel)
                edges.append(edge)

            # Remove any duplicate nodes (based on id)
            unique_nodes = {node.id: node for node in nodes}.values()

            return CitationNetwork(nodes=list(unique_nodes), edges=edges)

        except Exception as e:
            logger.error(f"Error processing relationships: {e}")
            return CitationNetwork(
                nodes=nodes[:1], edges=[]
            )  # Return at least the source node

    except Exception as e:
        logger.error(f"Error getting citation network: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting citation network: {str(e)}"
        )
