import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
import httpx
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)


# Models
class SearchResult(BaseModel):
    id: str
    name: str
    court_name: str
    date_filed: Optional[datetime] = None
    docket_number: Optional[str] = None
    citation: Optional[str] = None
    snippet: Optional[str] = None
    absolute_url: Optional[str] = None


class SearchResponse(BaseModel):
    count: int
    results: List[SearchResult]
    next: Optional[str] = None
    previous: Optional[str] = None


class RecentCase(BaseModel):
    id: str
    name: str
    court_name: str
    date_filed: Optional[datetime] = None
    docket_number: Optional[str] = None
    citation: Optional[str] = None
    snippet: Optional[str] = None


class RecentCasesResponse(BaseModel):
    count: int
    results: List[RecentCase]
    next: Optional[str] = None
    previous: Optional[str] = None


# CourtListener API endpoint
COURTLISTENER_API_URL = "https://www.courtlistener.com/api/rest/v4/search/"
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY", "")


# Search endpoint
@router.get("/search", response_model=SearchResponse)
async def search_cases(
    query: str,
    jurisdiction: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    court: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
):
    """
    Search for court cases using the CourtListener API.

    Parameters:
    - query: Search query string
    - jurisdiction: Filter by jurisdiction (e.g., "ca1", "ca2", etc.)
    - start_date: Filter by date filed (format: YYYY-MM-DD)
    - end_date: Filter by date filed (format: YYYY-MM-DD)
    - court: Filter by court (e.g., "scotus", "ca1", etc.)
    - page: Page number for pagination
    - page_size: Number of results per page
    """
    try:
        # Prepare query parameters
        params = {
            "q": query,
            "type": "o",  # opinions
            "page": page,
            "page_size": page_size,
        }

        # Add optional filters
        if jurisdiction:
            params["court"] = jurisdiction
        if start_date:
            params["filed_after"] = start_date
        if end_date:
            params["filed_before"] = end_date
        if court:
            params["court"] = court

        # Set up headers with API key if available
        headers = {}
        if COURTLISTENER_API_KEY:
            headers["Authorization"] = f"Token {COURTLISTENER_API_KEY}"

        # Make request to CourtListener API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                COURTLISTENER_API_URL, params=params, headers=headers, timeout=30.0
            )

            # Check for errors
            response.raise_for_status()
            data = response.json()

            # Transform response to our format
            results = []
            for item in data.get("results", []):
                result = SearchResult(
                    id=item.get("id", ""),
                    name=item.get("caseName", ""),
                    court_name=item.get("court", {}).get("fullName", ""),
                    date_filed=item.get("dateFiled"),
                    docket_number=item.get("docketNumber", ""),
                    citation=item.get("citation", ""),
                    snippet=item.get("snippet", ""),
                    absolute_url=item.get("absolute_url", ""),
                )
                results.append(result)

            return SearchResponse(
                count=data.get("count", 0),
                results=results,
                next=data.get("next"),
                previous=data.get("previous"),
            )

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during search: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error from CourtListener API: {e.response.text}",
        )
    except httpx.RequestError as e:
        logger.error(f"Request error during search: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error connecting to CourtListener API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# Recent cases endpoint
@router.get("/recent-cases", response_model=RecentCasesResponse)
async def get_recent_cases(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    Get a list of recent court cases.

    Parameters:
    - limit: Number of cases to return (default: 10, max: 100)
    - offset: Offset for pagination (default: 0)
    """
    try:
        # TODO: Implement Neo4j query to fetch recent cases
        # For now, we'll use the CourtListener API as a temporary solution

        # Make request to CourtListener API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                COURTLISTENER_API_URL,
                params={
                    "type": "o",  # opinions
                    "order_by": "-dateFiled",
                    "page_size": limit,
                    "page": (offset // limit) + 1 if limit > 0 else 1,
                },
                headers=(
                    {"Authorization": f"Token {COURTLISTENER_API_KEY}"}
                    if COURTLISTENER_API_KEY
                    else {}
                ),
                timeout=30.0,
            )

            # Check for errors
            response.raise_for_status()
            data = response.json()

            # Transform response to our format
            results = []
            for item in data.get("results", []):
                case = RecentCase(
                    id=item.get("id", ""),
                    name=item.get("caseName", ""),
                    court_name=item.get("court", {}).get("fullName", ""),
                    date_filed=item.get("dateFiled"),
                    docket_number=item.get("docketNumber", ""),
                    citation=item.get("citation", ""),
                    snippet=item.get("snippet", ""),
                )
                results.append(case)

            return RecentCasesResponse(
                count=data.get("count", 0),
                results=results,
                next=data.get("next"),
                previous=data.get("previous"),
            )

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching recent cases: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error from CourtListener API: {e.response.text}",
        )
    except httpx.RequestError as e:
        logger.error(f"Request error fetching recent cases: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error connecting to CourtListener API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching recent cases: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
