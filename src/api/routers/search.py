import logging
import os
import traceback
from datetime import datetime
from typing import List, Optional, Union

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.api.shared import templates

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
    ordering_key: Optional[str] = None
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
    page_size: Optional[int] = 20,
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
        # Log incoming request parameters
        logger.info(f"Search request params: {request.query_params}")
        
        # Convert empty strings to None for all parameters
        q = q if q and q.strip() else None
        jurisdiction = jurisdiction if jurisdiction and jurisdiction.strip() else None
        start_date = start_date if start_date and start_date.strip() else None
        end_date = end_date if end_date and end_date.strip() else None
        year_from = year_from if year_from and year_from.strip() else None
        year_to = year_to if year_to and year_to.strip() else None
        court = court if court and court.strip() and court != "all" else None
        order_by = order_by if order_by and order_by.strip() else None
        status = status if status and status.strip() else None
        highlight = highlight if highlight and highlight.strip() else None
        cursor = cursor if cursor and cursor.strip() else None
        
        # Ensure page_size is an integer
        try:
            page_size = int(page_size) if page_size is not None else 20
        except (ValueError, TypeError):
            logger.warning(f"Invalid page_size value: {page_size}, using default 20")
            page_size = 20
        
        # Check if we have any meaningful search criteria
        has_search_criteria = (
            q or jurisdiction or start_date or end_date or 
            year_from or year_to or court or status or cursor
        )
        
        # If no search criteria provided, return empty results
        if not has_search_criteria:
            empty_response = {
                "count": 0,
                "next": None,
                "previous": None,
                "results": []
            }
            
            # Check if the client wants HTML or JSON
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                # Return the rendered template with empty results
                return templates.TemplateResponse(
                    "components/search_results.html",
                    {"request": request, "results": {
                        "count": 0,
                        "next": None,
                        "previous": None,
                        "results": [],
                        "query": q
                    }}
                )
            
            # Return empty JSON response
            return SearchResponse(**empty_response)
        
        # Prepare query parameters
        params = {
            "type": type,
            "page_size": page_size,
        }

        # Add query if provided
        if q:
            params["q"] = q

        # Add optional filters
        if jurisdiction:
            params["court"] = jurisdiction
        if start_date:
            params["filed_after"] = start_date
        if end_date:
            params["filed_before"] = end_date

        # Handle year filters
        if year_from:
            try:
                year = int(year_from)
                params["filed_after"] = f"{year}-01-01"
            except ValueError:
                logger.warning(f"Invalid year_from value: {year_from}")
        if year_to:
            try:
                year = int(year_to)
                params["filed_before"] = f"{year}-12-31"
            except ValueError:
                logger.warning(f"Invalid year_to value: {year_to}")

        if court:
            params["court"] = court
        if order_by:
            params["order_by"] = order_by
        if status:
            params["status"] = status
        if highlight:
            params["highlight"] = highlight
        if cursor:
            params["cursor"] = cursor

        # Log the final parameters being sent to CourtListener
        logger.info(f"CourtListener API request params: {params}")
        logger.info(f"CourtListener API URL: {COURTLISTENER_API_URL}")

        # Set up headers with API key
        headers = {}
        if COURTLISTENER_API_KEY:
            headers["Authorization"] = f"Token {COURTLISTENER_API_KEY}"

        # Make request to CourtListener API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                COURTLISTENER_API_URL, params=params, headers=headers, timeout=30.0
            )

            # Log the response status and headers
            logger.info(f"CourtListener API response status: {response.status_code}")
            logger.debug(f"CourtListener API response headers: {response.headers}")
            
            # Check for errors
            try:
                response.raise_for_status()
                data = response.json()
                logger.debug(f"CourtListener API response data: {data}")
            except httpx.HTTPStatusError as e:
                logger.error(f"CourtListener API error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error parsing response: {str(e)}")
                logger.debug(f"Response content: {response.text}")
                raise

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
