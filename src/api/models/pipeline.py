"""
API models for pipeline operations.

This module refactors API models to use source-of-truth models from:
- src.llm_extraction.models
- src.neo4j.models 
- src.postgres.models

It uses them directly where possible and extends them where API-specific needs exist.
"""
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, date
from enum import Enum

# Import source-of-truth models
from src.llm_extraction.models import (
    CitationAnalysis, 
    CombinedResolvedCitationAnalysis, 
    Citation, 
    CitationResolved
)
from src.neo4j.neomodel_loader import NeomodelLoader

# Re-export source-of-truth models for API use
__all__ = [
    'CitationAnalysis', 
    'CombinedResolvedCitationAnalysis', 
    'Citation', 
    'CitationResolved',
    'JobStatus',
    'JobType',
    'ExtractionConfig',
    'PipelineJob',
    'PipelineStatus',
    'LLMProcessResult',
    'ResolutionResult',
    'Neo4jLoadResult',
    'PipelineResult',
    'PipelineStats'
]

class JobStatus(str, Enum):
    """Enum for pipeline job status."""
    QUEUED = "queued"
    STARTED = "started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobType(str, Enum):
    """Enum for pipeline job types."""
    EXTRACT = "extract"
    LLM_PROCESS = "llm_process"
    CITATION_RESOLUTION = "citation_resolution"
    NEO4J_LOAD = "neo4j_load"
    CSV_UPLOAD = "csv_upload"
    FULL_PIPELINE = "full_pipeline"

class ExtractionConfig(BaseModel):
    """Configuration for opinion extraction."""
    court_id: Optional[str] = Field(None, description="Filter by court ID")
    start_date: Optional[date] = Field(None, description="Filter by date range (start)")
    end_date: Optional[date] = Field(None, description="Filter by date range (end)")
    limit: Optional[int] = Field(None, description="Maximum number of opinions to extract")
    offset: Optional[int] = Field(0, description="Number of opinions to skip")
    include_text: bool = Field(True, description="Include opinion text in extraction")
    include_metadata: bool = Field(True, description="Include opinion metadata in extraction")
    
    @validator('limit')
    def validate_limit(cls, v):
        if v is not None and v <= 0:
            raise ValueError('limit must be positive')
        return v
    
    @validator('offset')
    def validate_offset(cls, v):
        if v < 0:
            raise ValueError('offset must be non-negative')
        return v

class PipelineJob(BaseModel):
    """Model for pipeline job information."""
    job_id: int = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Job status")
    message: Optional[str] = Field(None, description="Status message")

class PipelineStatus(BaseModel):
    """Model for pipeline job status."""
    job_id: int = Field(..., description="Job identifier")
    job_type: JobType = Field(..., description="Job type")
    status: JobStatus = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    config: Dict[str, Any] = Field(..., description="Job configuration")
    progress: Optional[float] = Field(None, description="Job progress (0-100)")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message if job failed")
    result_path: Optional[str] = Field(None, description="Path to job result file")
    
    model_config = ConfigDict(
        from_attributes=True  # Replaces deprecated orm_mode=True
    )

class LLMProcessResult(BaseModel):
    """Model for LLM processing result."""
    cluster_id: int = Field(..., description="Opinion cluster identifier")
    citation_analysis: CitationAnalysis = Field(..., description="Citation analysis result")
    
    @classmethod
    def from_gemini_response(cls, cluster_id: int, response: Dict[str, Any]):
        """Create LLMProcessResult from Gemini response."""
        import json
        from json_repair import repair_json
        
        citation_data = None
        if not cluster_id:
            raise ValueError("Cluster ID is required")
        
        # Handle the response based on its structure
        if "parsed" in response and response["parsed"] is not None:
            citation_data = response["parsed"]
            
            # If citation_data is still a string, parse it
            if isinstance(citation_data, str):
                try:
                    citation_data = json.loads(citation_data)
                except json.JSONDecodeError:
                    # Try to repair malformed JSON
                    citation_data = json.loads(repair_json(citation_data))
        else:
            # Try to extract from the candidate content
            try:
                raw_text = response["candidates"][0]["content"]["parts"][0]["text"]
                repaired_json = repair_json(raw_text)
                citation_data = json.loads(repaired_json)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                raise ValueError(f"Could not extract citation data from response: {str(e)}")
        
        # Validate and create CitationAnalysis
        citation_analysis = CitationAnalysis(
            date=citation_data["date"],
            brief_summary=citation_data["brief_summary"],
            majority_opinion_citations=citation_data.get("majority_opinion_citations", []),
            concurring_opinion_citations=citation_data.get("concurring_opinion_citations", []),
            dissenting_citations=citation_data.get("dissenting_citations", [])
        )
        
        return cls(cluster_id=cluster_id, citation_analysis=citation_analysis)

class ResolutionResult(BaseModel):
    """Model for citation resolution result."""
    cluster_id: int = Field(..., description="Opinion cluster identifier")
    resolved_citations: CombinedResolvedCitationAnalysis = Field(..., description="Resolved citation analysis")
    
    @classmethod
    def from_llm_result(cls, llm_result: LLMProcessResult):
        """Create ResolutionResult from LLMProcessResult."""
        from src.llm_extraction.models import CombinedResolvedCitationAnalysis
        
        # Use the built-in method from CombinedResolvedCitationAnalysis
        resolved = CombinedResolvedCitationAnalysis.from_citations(
            [llm_result.citation_analysis], 
            llm_result.cluster_id
        )
        
        return cls(
            cluster_id=llm_result.cluster_id,
            resolved_citations=resolved
        )

class Neo4jLoadResult(BaseModel):
    """Model for Neo4j loading result."""
    cluster_ids: List[int] = Field(..., description="Loaded opinion cluster IDs")
    citation_count: int = Field(..., description="Number of citations loaded")
    
    @classmethod
    def from_loader_result(cls, loader_result: Dict[str, Any]):
        """Create Neo4jLoadResult from loader result."""
        return cls(
            cluster_ids=loader_result.get("cluster_ids", []),
            citation_count=loader_result.get("citation_count", 0)
        )

class PipelineResult(BaseModel):
    """Model for pipeline job result."""
    job_id: int = Field(..., description="Job identifier")
    job_type: JobType = Field(..., description="Job type")
    status: JobStatus = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    result: Dict[str, Any] = Field(..., description="Job result data")
    
    model_config = ConfigDict(
        from_attributes=True  # Replaces deprecated orm_mode=True
    )

class PipelineStats(BaseModel):
    """Model for pipeline statistics."""
    total_jobs: int = Field(..., description="Total number of jobs")
    jobs_by_type: Dict[str, int] = Field(..., description="Jobs by type")
    jobs_by_status: Dict[str, int] = Field(..., description="Jobs by status")
    avg_processing_time: Dict[str, float] = Field(..., description="Average processing time by job type (seconds)")
    recent_jobs: List[PipelineStatus] = Field(..., description="Recent jobs")
