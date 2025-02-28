"""
API model definitions for the US Courts Legal Explorer API.

This package contains Pydantic models used for:
1. API request/response validation
2. Documentation generation
3. Data conversion between internal models and API responses

# Model Hierarchy and Organization

The system uses a layered model approach to reduce duplication:

## Core Models (Source of Truth)
- `src.llm_extraction.models`: Core data models for citations, opinions, and analysis
- `src.neo4j.models`: Graph database models using neomodel
- `src.postgres.models`: Relational database models using SQLAlchemy

## API Models (This Package)
- Directly import and re-export core models where possible
- Extend core models with API-specific fields when needed
- Provide conversion functions only when necessary

## Model Usage Guidelines
1. Always import models from their source-of-truth location
2. Use inheritance rather than duplication when extending models
3. Use the `from_*` class methods for conversion between model types
4. Avoid creating converter functions unless absolutely necessary

For example:
```python
# Good - import from source of truth
from src.llm_extraction.models import Citation

# Good - extend with inheritance
class APICitation(Citation):
    api_specific_field: str
    
    @classmethod
    def from_core(cls, citation: Citation):
        return cls(**citation.model_dump(), api_specific_field="value")
```
"""

from . import opinions, citations, stats, pipeline

# Export all modules
__all__ = ["opinions", "citations", "stats", "pipeline"]
