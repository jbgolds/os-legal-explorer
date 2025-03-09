import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Uncomment these imports for template and static file support
from fastapi.staticfiles import StaticFiles

# Import and include routers
from .services.pipeline import pipeline_router
from .services.batch_gemini.batch_gemini_router import router as batch_gemini_router
from .routers import search_router, clusters_router, feedback_router, network_router
from .routers.clusters import get_cluster_details, check_cluster_status
from .shared import templates
import logging

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
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://law.jbgolds.com",
    ],  # Allow the domain
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
app.include_router(clusters_router)
app.include_router(feedback_router)
app.include_router(network_router)
app.include_router(batch_gemini_router)


# Direct route for opinion pages
@app.get("/opinion/{cluster_id}/", response_class=HTMLResponse)
async def opinion_page(request: Request, cluster_id: str):
    try:
        # Use the existing case status endpoint logic
        case_status = await check_cluster_status(cluster_id)

        # Get case details from CourtListener API
        case_detail = await get_cluster_details(cluster_id)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "cluster_id": cluster_id,
                "case_status": case_status,
                "case": case_detail,
            },
        )
    except Exception as e:
        logger.error(f"Error loading case: {e}")
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": str(e)}
        )
