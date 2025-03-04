import logging
from typing import Optional, List, Dict, Any, Union
from fastapi import APIRouter, HTTPException, Query, Request, Depends
import httpx
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
from fastapi.templating import Jinja2Templates
import traceback

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


# Get templates from main app
def get_templates() -> Jinja2Templates:
    from ..main import templates

    return templates


# Models
class Opinion(BaseModel):
    author_id: Optional[Union[str, int]]
    cites: List[Union[str, int]] = []
    download_url: Optional[str]
    id: int
    joined_by_ids: List[Union[str, int]] = []
    local_path: Optional[str]
    meta: dict
    per_curiam: bool
    sha1: str
    snippet: str
    type: str


class SearchResult(BaseModel):
    absolute_url: str
    attorney: str
    caseName: str
    caseNameFull: str
    citation: List[str]
    citeCount: int
    cluster_id: int
    court: str
    court_citation_string: str
    court_id: str
    dateArgued: Optional[str]
    dateFiled: str
    dateReargued: Optional[str]
    dateReargumentDenied: Optional[str]
    docketNumber: str
    docket_id: int
    judge: str
    lexisCite: str
    meta: dict
    neutralCite: str
    non_participating_judge_ids: List[str] = []
    opinions: List[Opinion]
    panel_ids: List[str] = []
    panel_names: List[str] = []
    posture: str
    procedural_history: str
    scdb_id: str
    sibling_ids: List[int] = []
    source: str
    status: str
    suitNature: str
    syllabus: str


class SearchResponse(BaseModel):
    count: int
    next: Optional[str]
    previous: Optional[str]
    results: List[SearchResult]


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
COURTLISTENER_API_KEY = os.getenv(
    "COURTLISTENER_API_KEY", "***REMOVED***"
)


# Search endpoint
@router.get("/search")
async def search_cases(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    q: Optional[str] = None,
    type: str = "o",
    jurisdiction: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year_from: Optional[str] = None,
    year_to: Optional[str] = None,
    court: Optional[str] = None,
    order_by: Optional[str] = None,
    status: Optional[str] = None,
    highlight: Optional[str] = None,
    cursor: Optional[str] = None,
    page_size: int = 20,
):
    """
    Search for court cases using the CourtListener API v4.

    Parameters:
    - q: Search query string
    - type: Type of search (o=opinions, r=recap, rd=recap documents, d=dockets, p=people, oa=oral arguments)
    - jurisdiction: Filter by jurisdiction (e.g., "ca1", "ca2", etc.)
    - start_date: Filter by date filed (format: YYYY-MM-DD)
    - end_date: Filter by date filed (format: YYYY-MM-DD)
    - year_from: Filter by year (start)
    - year_to: Filter by year (end)
    - court: Filter by court (e.g., "scotus", "ca1", etc.)
    - order_by: Sort results (e.g., "score desc", "dateFiled desc")
    - status: Filter by status (e.g., "Published", "Unpublished")
    - highlight: Enable highlighting of results ("on" to enable)
    - cursor: Cursor for pagination
    - page_size: Number of results per page
    """
    try:
        # Prepare query parameters
        params = {
            "type": type,
            "page_size": page_size,
        }

        # Add query if provided
        if q and q.strip():
            params["q"] = q.strip()

        # Add optional filters
        if jurisdiction and jurisdiction.strip():
            params["court"] = jurisdiction.strip()
        if start_date and start_date.strip():
            params["filed_after"] = start_date.strip()
        if end_date and end_date.strip():
            params["filed_before"] = end_date.strip()

        # Handle year filters
        if year_from and year_from.strip():
            try:
                year = int(year_from)
                params["filed_after"] = f"{year}-01-01"
            except ValueError:
                pass
        if year_to and year_to.strip():
            try:
                year = int(year_to)
                params["filed_before"] = f"{year}-12-31"
            except ValueError:
                pass

        if court and court.strip() and court != "all":
            params["court"] = court.strip()
        if order_by and order_by.strip():
            params["order_by"] = order_by.strip()
        if status and status.strip():
            params["status"] = status.strip()
        if highlight and highlight.strip():
            params["highlight"] = highlight.strip()
        if cursor and cursor.strip():
            params["cursor"] = cursor.strip()

        # Set up headers with API key
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

            # Transform the data for our template
            transformed_data = {
                "count": data.get("count", 0),
                "next": data.get("next"),
                "previous": data.get("previous"),
                "results": [
                    {
                        "id": result.get("id"),
                        "case_name": result.get("caseName", ""),
                        "court_name": (
                            result.get("court", {}).get("fullName")
                            if isinstance(result.get("court"), dict)
                            else result.get("court", "Unknown Court")
                        ),
                        "date_filed": (
                            datetime.strptime(result["dateFiled"], "%Y-%m-%d").date()
                            if result.get("dateFiled")
                            else None
                        ),
                        "citation": result.get("citation", []),
                        "snippet": result.get("snippet", "No excerpt available."),
                        "cluster_id": result.get("cluster_id"),
                    }
                    for result in data.get("results", [])
                ],
                "query": q,
            }

            # Check if the client wants HTML (from HTMX) or JSON
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                # Return the rendered template
                return templates.TemplateResponse(
                    "components/search_results.html",
                    {"request": request, "results": transformed_data},
                )

            # Otherwise return JSON
            return SearchResponse(**data)

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
        logger.error(f"Unexpected error during search: {e}", traceback.format_exc())
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
