"""
This module has been refactored to remove duplicate citation model definitions.
All core citation-related models are now imported from src/llm_extraction.models.

You should now use the core models for citation information.

Note: If additional API-specific extensions are required, they can be added here.
"""

from typing import List
from pydantic import BaseModel
from src.llm_extraction.models import Citation, CitationAnalysis, CitationResolved


class CitationNode(BaseModel):
    """Represents a node in the citation network (an opinion)"""

    id: int
    title: str
    year: int
    court_id: str | None = None
    in_degree: int = 0
    out_degree: int = 0


class CitationEdge(BaseModel):
    """Represents an edge in the citation network (a citation relationship)"""

    source: int  # citing opinion id
    target: int  # cited opinion id
    weight: int = 1  # number of times cited


class CitationNetwork(BaseModel):
    """Represents a network of citations between opinions"""

    nodes: List[CitationNode]
    edges: List[CitationEdge]


class CitationDetail(BaseModel):
    """Detailed information about a specific citation relationship"""

    citing_id: int
    cited_id: int
    citing_title: str
    cited_title: str
    citation_context: str | None = None
    citation_type: str | None = None


class CitationStats(BaseModel):
    """Statistics about citations"""

    total_citations: int
    avg_citations_per_opinion: float
    most_cited_opinions: List[CitationDetail]


__all__ = [
    "Citation",
    "CitationAnalysis",
    "CitationResolved",
    "CitationNetwork",
    "CitationNode",
    "CitationEdge",
    "CitationDetail",
    "CitationStats",
]
