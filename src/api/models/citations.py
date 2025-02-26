from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import date

from src.neo4j.models import Opinion as Neo4jOpinion, CitesRel
from src.llm_extraction.models import OpinionSection, CitationTreatment, CitationResolved

class Node(BaseModel):
    """Model for a node in the citation network."""
    id: int = Field(..., description="Node identifier (cluster_id)")
    label: str = Field(..., description="Node label (case name)")
    type: str = Field("opinion", description="Node type")
    court_id: Optional[str] = Field(None, description="Court identifier")
    date_filed: Optional[date] = Field(None, description="Date the opinion was filed")
    citation_count: Optional[int] = Field(None, description="Number of times this opinion has been cited")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @classmethod
    def from_neo4j(cls, opinion: Neo4jOpinion, node_type: str = "opinion"):
        """Create a Node from a Neo4j Opinion model."""
        return cls(
            id=opinion.cluster_id,
            label=opinion.case_name or f"Opinion {opinion.cluster_id}",
            type=node_type,
            court_id=opinion.court_id,
            date_filed=opinion.date_filed,
            citation_count=len(opinion.cited_by.all()) if hasattr(opinion, 'cited_by') else None
        )

class Edge(BaseModel):
    """Model for an edge in the citation network."""
    source: int = Field(..., description="Source node identifier (citing opinion)")
    target: int = Field(..., description="Target node identifier (cited opinion)")
    treatment: Optional[str] = Field(None, description="Citation treatment (POSITIVE, NEGATIVE, etc.)")
    relevance: Optional[int] = Field(None, description="Relevance score (1-4)")
    opinion_section: Optional[str] = Field(None, description="Section of the opinion (majority, dissent, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @classmethod
    def from_neo4j_rel(cls, rel: CitesRel, source_id: int, target_id: int):
        """Create an Edge from a Neo4j CitesRel relationship."""
        return cls(
            source=source_id,
            target=target_id,
            treatment=rel.treatment,
            relevance=rel.relevance,
            opinion_section=rel.opinion_section,
            metadata={"source": rel.source}
        )

class CitationNetwork(BaseModel):
    """Model for a citation network."""
    nodes: List[Node] = Field(..., description="List of nodes in the network")
    edges: List[Edge] = Field(..., description="List of edges in the network")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Network metadata")

class CitationDetail(BaseModel):
    """Model for detailed citation information."""
    citing_opinion: Node = Field(..., description="Citing opinion")
    cited_opinion: Node = Field(..., description="Cited opinion")
    citation_text: str = Field(..., description="Original citation text")
    page_number: Optional[int] = Field(None, description="Page number where the citation appears")
    treatment: Optional[str] = Field(None, description="Citation treatment (POSITIVE, NEGATIVE, etc.)")
    relevance: Optional[int] = Field(None, description="Relevance score (1-4)")
    reasoning: Optional[str] = Field(None, description="Reasoning for the citation")
    opinion_section: Optional[str] = Field(None, description="Section of the opinion (majority, dissent, etc.)")
    source: Optional[str] = Field(None, description="Source of the citation data")
    
    @classmethod
    def from_citation_resolved(cls, citation: CitationResolved, citing_opinion: Neo4jOpinion, cited_opinion: Neo4jOpinion):
        """Create a CitationDetail from a CitationResolved and Neo4j Opinion models."""
        return cls(
            citing_opinion=Node.from_neo4j(citing_opinion),
            cited_opinion=Node.from_neo4j(cited_opinion),
            citation_text=citation.citation_text,
            page_number=citation.page_number,
            treatment=citation.treatment,
            relevance=citation.relevance,
            reasoning=citation.reasoning,
            opinion_section=OpinionSection.majority.value,  # Default to majority if not specified
            source="llm_extraction"
        )
    
    @classmethod
    def from_neo4j_rel(cls, rel: CitesRel, citing_opinion: Neo4jOpinion, cited_opinion: Neo4jOpinion):
        """Create a CitationDetail from a Neo4j CitesRel relationship and Opinion models."""
        return cls(
            citing_opinion=Node.from_neo4j(citing_opinion),
            cited_opinion=Node.from_neo4j(cited_opinion),
            citation_text=rel.citation_text or f"Citation from {citing_opinion.case_name} to {cited_opinion.case_name}",
            page_number=rel.page_number,
            treatment=rel.treatment,
            relevance=rel.relevance,
            reasoning=rel.reasoning,
            opinion_section=rel.opinion_section,
            source=rel.source
        )

class CitationStats(BaseModel):
    """Model for citation statistics."""
    total_citations: int = Field(..., description="Total number of citations")
    total_opinions: int = Field(..., description="Total number of opinions")
    avg_citations_per_opinion: float = Field(..., description="Average citations per opinion")
    treatment_counts: Dict[str, int] = Field(..., description="Citation treatment counts")
    relevance_distribution: Dict[int, int] = Field(..., description="Citation relevance distribution")
    section_distribution: Dict[str, int] = Field(..., description="Citation section distribution")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
