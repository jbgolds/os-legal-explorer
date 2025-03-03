# This file marks the directory as a Python package
# Import and expose the pipeline_router
from .pipeline_router import router as pipeline_router

__all__ = ["pipeline_router"]
