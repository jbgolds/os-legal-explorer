import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Uncomment these imports for template and static file support
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import date

# Import and include routers
from .services.pipeline import pipeline_router
from .routers import search_router, cases_router, feedback_router

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

# Configure templates and static files
templates = Jinja2Templates(directory="src/frontend/templates")


# Add custom filters
def format_date(value):
    if isinstance(value, date):
        return value.strftime("%b %d, %Y")
    return value


templates.env.filters["date"] = format_date

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
app.include_router(cases_router)
app.include_router(feedback_router)


# Case detail page route
@app.get("/case/{case_id}", response_class=HTMLResponse)
async def case_detail(request: Request, case_id: str):
    return templates.TemplateResponse(
        "case_detail.html", {"request": request, "case_id": case_id}
    )
