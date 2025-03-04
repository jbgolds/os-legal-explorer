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
    tags=["clusters"],
    responses={404: {"description": "Not found"}},
)


# Models
class Citation(BaseModel):
    id: str
    name: str
    citation: Optional[str] = None
    treatment: Optional[str] = None  # positive, negative, neutral, etc.


class Cluster(BaseModel):
    cluster_id: str
    case_name: str
    court_name: str
    date_filed: datetime
    citation: Optional[str] = None
    html: Optional[str] = None
    plain_text: Optional[str] = None


class ClusterDetail(BaseModel):
    id: str
    name: str
    court_name: str
    date_filed: Optional[datetime] = None
    docket_number: Optional[str] = None
    citation: Optional[str] = None
    judges: Optional[List[str]] = None
    opinion_text: Optional[str] = None
    citations: Optional[List[Citation]] = None


class ClusterStatus(BaseModel):
    exists: bool
    has_citations: bool
    citation_count: Optional[int] = None


# CourtListener API endpoint
COURTLISTENER_API_URL = "https://www.courtlistener.com/api/rest/v4"
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY", "")


@router.get("/{cluster_id}", response_model=ClusterDetail)
async def get_cluster_details(cluster_id: str) -> Optional[ClusterDetail]:
    """
    Get cluster details from CourtListener API.

    Parameters:
    - cluster_id: The ID of the cluster to get details for

    Returns:
    - ClusterDetail object or None if not found
    """
    try:
        # Get API token from environment variables
        api_token = os.getenv("COURTLISTENER_API_KEY")
        if not api_token:
            logger.error("CourtListener API token not found in environment variables")
            return None

        # Make request to CourtListener API
        headers = {"Authorization": f"Token {api_token}"}

        # Get cluster details
        async with httpx.AsyncClient() as client:
            # Get cluster details
            response = await client.get(
                f"https://www.courtlistener.com/api/rest/v4/clusters/{cluster_id}/",
                headers=headers,
                timeout=10.0,
            )

            if response.status_code == 404:
                logger.warning(
                    f"Cluster with cluster_id {cluster_id} not found in CourtListener API"
                )
                return None

            if response.status_code != 200:
                logger.error(
                    f"Error from CourtListener API: {response.status_code} - {response.text}"
                )
                return None

            data = response.json()

            # Extract citation if available
            citation = None
            if data.get("citations") and len(data["citations"]) > 0:
                citation_obj = data["citations"][0]
                if all(k in citation_obj for k in ["volume", "reporter", "page"]):
                    citation = f"{citation_obj['volume']} {citation_obj['reporter']} {citation_obj['page']}"

            # Extract judges if available
            judges = []
            if data.get("judges"):
                judges = [j.strip() for j in data["judges"].split(",") if j.strip()]

            # Get opinion text if available
            opinion_text = None
            if data.get("sub_opinions") and len(data["sub_opinions"]) > 0:
                # Get the first opinion's text
                opinion_url = data["sub_opinions"][0]
                if isinstance(opinion_url, str) and opinion_url.startswith("http"):
                    opinion_response = await client.get(
                        opinion_url, headers=headers, timeout=10.0
                    )

                    if opinion_response.status_code == 200:
                        opinion_data = opinion_response.json()
                        opinion_text = opinion_data.get("plain_text", "")
                        if not opinion_text:
                            opinion_text = opinion_data.get("html", "")
                elif isinstance(opinion_url, dict) and "id" in opinion_url:
                    # Handle cluster where sub_opinions contains objects instead of URLs
                    opinion_id = opinion_url["id"]
                    opinion_response = await client.get(
                        f"https://www.courtlistener.com/api/rest/v4/opinions/{opinion_id}/",
                        headers=headers,
                        timeout=10.0,
                    )

                    if opinion_response.status_code == 200:
                        opinion_data = opinion_response.json()
                        opinion_text = opinion_data.get("plain_text", "")

            # Get docket information
            docket_number = None
            if isinstance(data.get("docket"), str) and data["docket"].startswith(
                "http"
            ):
                # If docket is a URL, fetch the docket details
                docket_response = await client.get(
                    data["docket"], headers=headers, timeout=10.0
                )

                if docket_response.status_code == 200:
                    docket_data = docket_response.json()
                    docket_number = docket_data.get("docket_number")
            elif isinstance(data.get("docket"), dict):
                # If docket is an object, extract docket_number directly
                docket_number = data["docket"].get("docket_number")

            # Get court information
            court_name = "Unknown Court"
            if isinstance(data.get("court"), str) and data["court"].startswith("http"):
                # If court is a URL, fetch the court details
                court_response = await client.get(
                    data["court"], headers=headers, timeout=10.0
                )

                if court_response.status_code == 200:
                    court_data = court_response.json()
                    court_name = court_data.get("full_name", "Unknown Court")
            elif isinstance(data.get("court"), dict):
                # If court is an object, extract full_name directly
                court_name = data["court"].get("full_name", "Unknown Court")

            # Create ClusterDetail object from API response
            cluster_detail = ClusterDetail(
                id=str(data["id"]),
                name=data.get("case_name", "Unknown Cluster"),
                court_name=court_name,
                date_filed=(
                    datetime.fromisoformat(data["date_filed"])
                    if data.get("date_filed")
                    else None
                ),
                docket_number=docket_number,
                citation=citation,
                judges=judges,
                opinion_text=opinion_text,
                citations=[],  # We'll skip fetching citations for now to simplify the implementation
            )

            return cluster_detail

    except Exception as e:
        logger.error(f"Error getting cluster details from CourtListener API: {e}")
        return None


@router.get("/{cluster_id}/status")
async def check_cluster_status(cluster_id: str) -> ClusterStatus:
    """
    Check if a cluster exists in Neo4j and has outgoing citations.

    Parameters:
    - cluster_id: The ID of the cluster to check

    Returns:
    - ClusterStatus object with existence and citation information
    """
    try:
        # Try to find the opinion by primary_id (cluster_id)
        opinion = Opinion.nodes.first_or_none(primary_id=cluster_id)

        if not opinion:
            return ClusterStatus(exists=False, has_citations=False)

        # Count outgoing citations in a separate transaction
        with db.transaction:
            # Execute Cypher query to count outgoing CITES relationships
            query = """
            MATCH (n:Opinion {primary_id: $primary_id})-[r:CITES]->()
            RETURN count(r) as citation_count
            """
            results, _ = db.cypher_query(query, {"primary_id": cluster_id})
            citation_count = results[0][0] if results and results[0] else 0

        return ClusterStatus(
            exists=True, has_citations=citation_count > 0, citation_count=citation_count
        )

    except Exception as e:
        logger.error(f"Error checking cluster status: {e}")
        # Return "not found" status on any error
        return ClusterStatus(exists=False, has_citations=False)


# Opinion page route - serves the same index.html but with pre-loaded cluster data
@router.get("/{cluster_id}/", response_class=HTMLResponse)
async def opinion(request: Request, cluster_id: str):
    try:
        # Use the existing cluster status endpoint logic
        cluster_status = await check_cluster_status(cluster_id)

        # Get cluster details from CourtListener API
        cluster_detail = await get_cluster_details(cluster_id)

        return templates.TemplateResponse(
            "opinion.html",
            {
                "request": request,
                "cluster_id": cluster_id,
                "cluster_status": cluster_status,
                "cluster": cluster_detail,
            },
        )
    except Exception as e:
        logger.error(f"Error loading cluster: {e}")
        return templates.TemplateResponse(
            "opinion.html", {"request": request, "error": str(e)}
        )
