"""
API models for statistics.

This module refactors API models to use source-of-truth models from:
- src.llm_extraction.models
- src.neo4j.models 
- src.postgres.models

It uses them directly where possible and extends them where API-specific needs exist.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import date

from src.neo4j.models import Opinion as Neo4jOpinion

__all__ = [
    'NetworkStats',
    'CourtStats',
    'YearlyStats',
    'TimelineStats',
    'TopCitedOpinion',
    'TopCitedOpinions'
]

class NetworkStats(BaseModel):
    """Model for overall network statistics."""
    total_nodes: int = Field(..., description="Total number of nodes in the network")
    total_edges: int = Field(..., description="Total number of edges in the network")
    network_density: float = Field(..., description="Network density (ratio of actual to possible connections)")
    avg_degree: float = Field(..., description="Average node degree (number of connections)")
    max_degree: int = Field(..., description="Maximum node degree")
    avg_path_length: Optional[float] = Field(None, description="Average shortest path length")
    diameter: Optional[int] = Field(None, description="Network diameter (longest shortest path)")
    clustering_coefficient: Optional[float] = Field(None, description="Global clustering coefficient")
    connected_components: Optional[int] = Field(None, description="Number of connected components")
    largest_component_size: Optional[int] = Field(None, description="Size of the largest connected component")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class CourtStats(BaseModel):
    """Model for court-specific statistics."""
    court_id: str = Field(..., description="Court identifier")
    court_name: str = Field(..., description="Court name")
    opinion_count: int = Field(..., description="Number of opinions from this court")
    citation_count: int = Field(..., description="Number of citations from this court")
    cited_by_count: int = Field(..., description="Number of times opinions from this court are cited")
    self_citation_ratio: float = Field(..., description="Ratio of citations to the same court")
    avg_citations_per_opinion: float = Field(..., description="Average citations per opinion")
    top_cited_courts: List[Dict[str, Any]] = Field(..., description="Top courts cited by this court")
    top_citing_courts: List[Dict[str, Any]] = Field(..., description="Top courts citing this court")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class YearlyStats(BaseModel):
    """Model for yearly statistics."""
    year: int = Field(..., description="Year")
    opinion_count: int = Field(..., description="Number of opinions in this year")
    citation_count: int = Field(..., description="Number of citations in this year")
    cited_by_count: int = Field(..., description="Number of times opinions from this year are cited")
    avg_citations_per_opinion: float = Field(..., description="Average citations per opinion")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class TimelineStats(BaseModel):
    """Model for timeline statistics."""
    yearly_stats: List[YearlyStats] = Field(..., description="Statistics by year")
    total_years: int = Field(..., description="Total number of years")
    start_year: int = Field(..., description="First year in the timeline")
    end_year: int = Field(..., description="Last year in the timeline")

class TopCitedOpinion(BaseModel):
    """Model for a top cited opinion."""
    cluster_id: int = Field(..., description="Opinion cluster identifier")
    case_name: str = Field(..., description="Case name")
    court_id: str = Field(..., description="Court identifier")
    court_name: str = Field(..., description="Court name")
    date_filed: date = Field(..., description="Date the opinion was filed")
    citation_count: int = Field(..., description="Number of times this opinion has been cited")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @classmethod
    def from_neo4j(cls, opinion: Neo4jOpinion, citation_count: int):
        """Create a TopCitedOpinion from a Neo4j Opinion model and citation count."""
        return cls(
            cluster_id=opinion.cluster_id,
            case_name=opinion.case_name or f"Opinion {opinion.cluster_id}",
            court_id=str(opinion.court_id),
            court_name=opinion.court_name or "Unknown Court",
            date_filed=opinion.date_filed,
            citation_count=citation_count,
            metadata={
                "docket_number": opinion.docket_number,
                "brief_summary": opinion.ai_summary
            }
        )
    
    model_config = ConfigDict(
        from_attributes=True  # Replaces deprecated orm_mode=True
    )

class TopCitedOpinions(BaseModel):
    """Model for top cited opinions."""
    opinions: List[TopCitedOpinion] = Field(..., description="List of top cited opinions")
    total_count: int = Field(..., description="Total number of opinions in the database")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
