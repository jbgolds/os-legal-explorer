import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import and include routers
from .routers import opinions, citations, stats, pipeline
from .views.views import router as views_router

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
app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")


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
app.include_router(opinions.router)
app.include_router(citations.router)
app.include_router(stats.router)
app.include_router(pipeline.router)
app.include_router(views_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True
    )
