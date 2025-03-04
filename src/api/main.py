import os
import logging
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Uncomment these imports for template and static file support
from fastapi.staticfiles import StaticFiles
from datetime import date, datetime

# Import and include routers
from .services.pipeline import pipeline_router
from .routers import search_router, cases_router, feedback_router
from .routers.cases import Case, check_case_status
from .shared import templates
import logging
from src.neo4j_db.models import Opinion

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure simple logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI(
    title="OS Legal Explorer API",
    description="API for exploring legal citation networks",
    version="0.1.0",
)

# Configure CORS - allows all origins for simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Simple setting for development/hobby project
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure static files
app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")


# Home page route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Opinion page route - serves the same index.html but with pre-loaded case data
@app.get("/opinion/{cluster_id}/", response_class=HTMLResponse)
async def opinion(request: Request, cluster_id: str):
    try:
        # Use the existing case status endpoint logic
        case_status = await check_case_status(cluster_id)

        # Get opinion from Neo4j if it exists
        opinion = (
            Opinion.nodes.first_or_none(primary_id=cluster_id)
            if case_status.exists
            else None
        )

        # Create case object with available data
        case = Case(
            cluster_id=cluster_id,
            case_name=opinion.case_name if opinion else "Sample Case Name",
            court_name=opinion.court_name if opinion else "Sample Court",
            date_filed=opinion.date_filed if opinion else datetime.now(),
            citation=opinion.citation_string if opinion else None,
            plain_text="Sample case text...",  # TODO: Get from PostgreSQL
        )

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "preloaded_case": case,
                "preloaded_cluster_id": cluster_id,
                "case_status": case_status,
            },
        )
    except Exception as e:
        logger.error(f"Error loading case: {e}")
        return templates.TemplateResponse("index.html", {"request": request})


# Health check endpoint
@app.get("/health")
async def health_check():
    from .database import verify_connections

    db_status = verify_connections()
    return {
        "status": "healthy" if all(db_status.values()) else "unhealthy",
        "database": db_status,
    }


# Include all routers
app.include_router(pipeline_router)
app.include_router(search_router)
app.include_router(cases_router)
app.include_router(feedback_router)


# Case detail page route
@app.get("/case/{case_id}", response_class=HTMLResponse)
async def case_detail(request: Request, case_id: str):
    return templates.TemplateResponse(
        "case_detail.html", {"request": request, "case_id": case_id}
    )
