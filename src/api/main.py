from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="OS Legal Explorer API",
    description="API for exploring legal citation networks",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
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
    return {"status": "healthy"}

# Import and include routers
from .routers import opinions, citations, stats, pipeline
app.include_router(opinions.router)
app.include_router(citations.router)
app.include_router(stats.router)
app.include_router(pipeline.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
