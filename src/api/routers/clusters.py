import logging
import os
from datetime import datetime
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from neomodel import adb, db
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text

from src.api.shared import templates
from src.neo4j_db.models import Opinion
from src.neo4j_db.neomodel_loader import NeomodelLoader
from src.postgres.database import get_db_session

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
    case_name: str
    court_name: str
    date_filed: Optional[datetime] = None
    docket_number: Optional[str] = None
    citation: Optional[str] = None
    judges: Optional[List[str]] = None
    opinion_text: Optional[str] = None
    citations: Optional[List[Citation]] = None
    ai_summary: Optional[str] = None
    download_url: Optional[str] = None


class ClusterStatus(BaseModel):
    exists: bool
    has_citations: bool
    citation_count: Optional[int] = None
    has_ai_summary: bool = False


# CourtListener API endpoint
COURTLISTENER_API_URL = "https://www.courtlistener.com/api/rest/v4"
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY", "")


@router.get("/{cluster_id}", response_model=ClusterDetail)
async def get_cluster_details(cluster_id: str) -> Optional[ClusterDetail]:
    """
    Get cluster details primarily from PostgreSQL database, with supplementary citation data from Neo4j.
    Falls back to CourtListener API if the cluster is not found in our local database.

    Parameters:
    - cluster_id: The ID of the cluster to get details for

    Returns:
    - ClusterDetail object or None if not found
    """
    try:
        # Initialize variables
        name = "Unknown Cluster"
        court_name = "Unknown Court"
        date_filed = None
        docket_number = None
        citation = None
        judges = []
        opinion_text = None
        citations = []

        # First, query PostgreSQL for the primary data
        try:
            with get_db_session() as db_session:
                # Get cluster details
                cluster_query = text(
                    """
                SELECT 
                    soc.case_name, 
                    sc.full_name as court_name,
                    soc.date_filed,
                    sd.docket_number,
                    soc.judges
                FROM search_opinioncluster soc
                JOIN search_docket sd ON soc.docket_id = sd.id
                JOIN search_court sc ON sd.court_id = sc.id
                WHERE soc.id = :cluster_id
                """
                )

                cluster_result = db_session.execute(
                    cluster_query, {"cluster_id": int(cluster_id)}
                ).fetchone()

                if cluster_result:
                    name = cluster_result[0] or name
                    court_name = cluster_result[1] or court_name
                    date_filed = cluster_result[2]
                    docket_number = cluster_result[3]

                    # Extract judges if available
                    if cluster_result[4]:
                        judges = [
                            j.strip() for j in cluster_result[4].split(",") if j.strip()
                        ]
                else:
                    # If not found in PostgreSQL, we'll try CourtListener API later
                    logger.warning(
                        f"Cluster {cluster_id} not found in PostgreSQL database"
                    )
                    raise ValueError("Cluster not found in PostgreSQL")

                # Get opinion text
                opinion_text_query = text(
                    """
                SELECT download_url, html_with_citations, html, plain_text
                FROM search_opinion
                WHERE cluster_id = :cluster_id
                ORDER BY 
                    CASE 
                        WHEN type = '010combined' THEN 0
                        WHEN type = '020lead' THEN 1
                        ELSE 2
                    END,
                    id
                LIMIT 1
                """
                )

                opinion_result = db_session.execute(
                    opinion_text_query, {"cluster_id": int(cluster_id)}
                ).fetchone()

                if opinion_result:
                    # Prioritize html_with_citations, then html, then plain_text
                    opinion_text = (
                        opinion_result[1] or opinion_result[2] or opinion_result[3]
                    )
                    download_url = opinion_result[0]
                    

                # Get citation
                citation_query = text(
                    """
                SELECT volume, reporter, page
                FROM search_citation
                WHERE cluster_id = :cluster_id
                LIMIT 1
                """
                )

                citation_result = db_session.execute(
                    citation_query, {"cluster_id": int(cluster_id)}
                ).fetchone()

                if citation_result and all(citation_result):
                    citation = f"{citation_result[0]} {citation_result[1]} {citation_result[2]}"

                # Now that we have the basic data, check Neo4j for citation relationships
                try:
                    # Check if the opinion exists in Neo4j
                    opinion = await Opinion.nodes.first_or_none(primary_id=cluster_id)

                    # If it doesn't exist, we might want to create it for future use
                    if not opinion and name != "Unknown Cluster":
                        # Create a new opinion node with the data we have
                        opinion = Opinion(
                            primary_id=str(cluster_id),
                            case_name=name,
                            court_name=court_name,
                            date_filed=date_filed,
                            docket_number=docket_number,
                            citation_string=citation,
                        )
                        await opinion.save()
                        logger.info(
                            f"Created new opinion node in Neo4j for cluster {cluster_id}"
                        )

                    # If we have an opinion node (existing or newly created), fetch its citations
                    if opinion:
                        try:
                            # Use the neomodel relationship traversal instead of raw Cypher
                            cited_opinions = await opinion.cites.all()

                            # Transform results into Citation objects
                            for cited in cited_opinions:
                                # Get the relationship properties
                                rel_props = await opinion.cites.relationship(cited)
                                treatment = (
                                    rel_props.treatment
                                    if hasattr(rel_props, "treatment")
                                    else None
                                )

                                # Get citation name with fallback
                                const_cited_name = (
                                    (hasattr(cited, "case_name") and cited.case_name)
                                    or cited.title
                                    or "Unknown Case"
                                )
                                citations.append(
                                    Citation(
                                        id=cited.primary_id or "",
                                        name=const_cited_name,
                                        citation=cited.citation_string,
                                        treatment=treatment,
                                    )
                                )
                        except Exception as e:
                            logger.error(f"Error fetching citations with neomodel: {e}")
                            # Continue with what we have from PostgreSQL
                except Exception as e:
                    logger.error(f"Error with Neo4j operations: {e}")
                    # Continue with what we have from PostgreSQL

                # Create ClusterDetail object from the data we've gathered
                cluster_detail = ClusterDetail(
                    id=str(cluster_id),
                    case_name=name,
                    court_name=court_name,
                    date_filed=date_filed,
                    docket_number=docket_number,
                    citation=citation,
                    judges=judges,
                    opinion_text=opinion_text,
                    citations=citations,
                    ai_summary=getattr(opinion, "ai_summary", None) if opinion else None,
                    download_url=download_url if 'download_url' in locals() else None,
                )

                return cluster_detail

        except (SQLAlchemyError, ValueError) as e:
            logger.error(f"Database error when fetching cluster details: {e}")
            # Fall back to CourtListener API

        # If we get here, we need to fall back to CourtListener API
        logger.warning(f"Falling back to CourtListener API for cluster {cluster_id}")

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
            logger.debug(f"CourtListener response data: {data}")

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

                        if not opinion_text:
                            opinion_text = opinion_data.get("html_with_citations", "")

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
                case_name=data.get("caseName", "Unknown Case"),
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
                ai_summary=getattr(opinion, "ai_summary", None) if opinion else None,
                download_url=None,  # CourtListener API doesn't provide direct PDF download links
            )

            # Try to create a Neo4j node for this cluster for future use
            try:
                opinion = await Opinion.nodes.first_or_none(primary_id=cluster_id)
                if not opinion:
                    opinion = Opinion(
                        primary_id=str(cluster_id),
                        primary_table="opinion_cluster",
                        case_name=data.get("caseName", "Unknown Case"),
                        court_name=court_name,
                        date_filed=(
                            datetime.fromisoformat(data["date_filed"])
                            if data.get("date_filed")
                            else None
                        ),
                        docket_number=docket_number,
                        citation_string=citation,
                    )
                    await opinion.save()
                    logger.info(
                        f"Created new opinion node in Neo4j for cluster {cluster_id} from CourtListener API"
                    )
            except Exception as e:
                logger.error(f"Error creating Neo4j node from CourtListener data: {e}")
                # Continue with the response

            return cluster_detail

    except Exception as e:
        logger.error(f"Error getting cluster details: {e}")
        return None


@router.get("/{cluster_id}/status")
async def check_cluster_status(cluster_id: str) -> ClusterStatus:
    """
    Check if a cluster exists in Neo4j, has outgoing citations, and has ai_summary.

    Parameters:
    - cluster_id: The ID of the cluster to check

    Returns:
    - ClusterStatus object with existence, citation, and ai_summary information
    """
    try:
        # Try to find the opinion by primary_id (cluster_id)
        opinion = await Opinion.nodes.first_or_none(primary_id=cluster_id)

        if not opinion:
            return ClusterStatus(
                exists=False, has_citations=False, has_ai_summary=False
            )

        # Check if ai_summary is filled out
        has_ai_summary = bool(opinion.ai_summary)

        # Count outgoing citations in a separate transaction
        with db.transaction:
            # Execute Cypher query to count outgoing CITES relationships
            query = """
            MATCH (n:Opinion {primary_id: $primary_id})-[r:CITES]->()
            RETURN count(r) as citation_count
            """
            results, _ = await adb.cypher_query(query, {"primary_id": cluster_id})
            citation_count = results[0][0] if results and results[0] else 0

        return ClusterStatus(
            exists=True,
            has_citations=citation_count > 0,
            citation_count=citation_count,
            has_ai_summary=has_ai_summary,
        )

    except Exception as e:
        logger.error(f"Error checking cluster status: {e}")
        # Return "not found" status on any error
        return ClusterStatus(exists=False, has_citations=False, has_ai_summary=False)
