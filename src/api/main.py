import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure simple logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the OS Legal Explorer API",
        "docs": "/docs",
        "version": "0.1.0",
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    from .database import verify_connections
    db_status = verify_connections()
    return {
        "status": "healthy" if all(db_status.values()) else "unhealthy",
        "database": db_status,
    }

# Import and include routers
from .routers import opinions, citations, stats, pipeline
app.include_router(opinions.router)
app.include_router(citations.router)
app.include_router(stats.router)
app.include_router(pipeline.router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
