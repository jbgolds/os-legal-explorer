from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

from src.neo4j.models import Opinion as Neo4jOpinion
from src.llm_extraction.models import OpinionSection, CitationTreatment

class OpinionBase(BaseModel):
    """Base model for opinion data."""
    cluster_id: int = Field(..., description="Unique identifier for the opinion cluster")
    case_name: str = Field(..., description="Name of the case")
    date_filed: date = Field(..., description="Date the opinion was filed")
    
    @classmethod
    def from_neo4j(cls, opinion: Neo4jOpinion):
        """Create an OpinionBase from a Neo4j Opinion model."""
        return cls(
            cluster_id=opinion.cluster_id,
            case_name=opinion.case_name,
            date_filed=opinion.date_filed
        )
    
class OpinionResponse(OpinionBase):
    """Model for opinion summary response."""
    court_id: str = Field(..., description="Court identifier")
    court_name: str = Field(..., description="Name of the court")
    docket_number: Optional[str] = Field(None, description="Docket number")
    citation_count: int = Field(0, description="Number of times this opinion has been cited")
    
    @classmethod
    def from_neo4j(cls, opinion: Neo4jOpinion):
        """Create an OpinionResponse from a Neo4j Opinion model."""
        return cls(
            cluster_id=opinion.cluster_id,
            case_name=opinion.case_name,
            date_filed=opinion.date_filed,
            court_id=opinion.court_id,
            court_name=opinion.court_name or "Unknown Court",
            docket_number=opinion.docket_number,
            citation_count=len(opinion.cited_by.all()) if hasattr(opinion, 'cited_by') else 0
        )
    
    class Config:
        orm_mode = True

class OpinionDetail(OpinionResponse):
    """Model for detailed opinion information."""
    docket_id: Optional[int] = Field(None, description="Docket identifier")
    scdb_votes_majority: Optional[int] = Field(None, description="Supreme Court Database majority votes")
    scdb_votes_minority: Optional[int] = Field(None, description="Supreme Court Database minority votes")
    brief_summary: Optional[str] = Field(None, description="Brief summary of the opinion")
    
    @classmethod
    def from_neo4j(cls, opinion: Neo4jOpinion):
        """Create an OpinionDetail from a Neo4j Opinion model."""
        base = OpinionResponse.from_neo4j(opinion)
        return cls(
            **base.dict(),
            docket_id=opinion.docket_id,
            scdb_votes_majority=opinion.scdb_votes_majority,
            scdb_votes_minority=opinion.scdb_votes_minority,
            brief_summary=opinion.ai_summary
        )
    
    class Config:
        orm_mode = True

class OpinionText(BaseModel):
    """Model for opinion text."""
    cluster_id: int = Field(..., description="Opinion cluster identifier")
    text: str = Field(..., description="Full text of the opinion")
    
    class Config:
        orm_mode = True

class OpinionCitation(BaseModel):
    """Model for citation information within an opinion."""
    citing_id: int = Field(..., description="Citing opinion cluster identifier")
    cited_id: int = Field(..., description="Cited opinion cluster identifier")
    citation_text: str = Field(..., description="Original citation text")
    page_number: Optional[int] = Field(None, description="Page number where the citation appears")
    treatment: Optional[str] = Field(None, description="Citation treatment (POSITIVE, NEGATIVE, etc.)")
    relevance: Optional[int] = Field(None, description="Relevance score (1-4)")
    reasoning: Optional[str] = Field(None, description="Reasoning for the citation")
    opinion_section: Optional[str] = Field(None, description="Section of the opinion (majority, dissent, etc.)")
    
    @classmethod
    def from_neo4j_rel(cls, rel, citing_id: int, cited_id: int):
        """Create an OpinionCitation from a Neo4j CitesRel relationship."""
        return cls(
            citing_id=citing_id,
            cited_id=cited_id,
            citation_text=rel.citation_text or "",
            page_number=rel.page_number,
            treatment=rel.treatment,
            relevance=rel.relevance,
            reasoning=rel.reasoning,
            opinion_section=rel.opinion_section
        )
    
    class Config:
        orm_mode = True
