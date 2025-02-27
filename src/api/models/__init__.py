"""
API model definitions for the US Courts Legal Explorer API.

This package contains Pydantic models used for:
1. API request/response validation
2. Documentation generation
3. Data conversion between internal models and API responses

Note: Many models now directly use or extend core models from src.llm_extraction.models,
src.neo4j.models, and src.postgres.models to reduce duplication.
"""

from . import opinions, citations, stats, pipeline, converters

# Export all modules
__all__ = ["opinions", "citations", "stats", "pipeline", "converters"]
