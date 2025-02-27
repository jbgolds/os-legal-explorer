"""
This module has been refactored to remove duplicate citation model definitions.
All core citation-related models are now imported from src/llm_extraction.models.

You should now use the core models for citation information.

Note: If additional API-specific extensions are required, they can be added here.
"""

from src.llm_extraction.models import Citation, CitationAnalysis, CitationResolved

__all__ = ['Citation', 'CitationAnalysis', 'CitationResolved']
