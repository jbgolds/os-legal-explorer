"""
API models for pipeline operations.

This module refactors API models to use source-of-truth models from:
- src.llm_extraction.models
- src.neo4j.models 
- src.postgres.models

It uses them directly where possible and extends them where API-specific needs exist.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, validator

# Import source-of-truth models
from src.llm_extraction.models import (CitationAnalysis,
                                       CombinedResolvedCitationAnalysis)


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
    limit: Optional[int] = Field(
        None, description="Maximum number of opinions to extract"
    )
    offset: Optional[int] = Field(0, description="Number of opinions to skip")
    include_text: bool = Field(True, description="Include opinion text in extraction")
    include_metadata: bool = Field(
        True, description="Include opinion metadata in extraction"
    )
    single_cluster_id: Optional[int] = Field(
        None, description="Process a single cluster by ID"
    )

    @validator("limit")
    def validate_limit(cls, v):
        if v is not None and v <= 0:
            raise ValueError("limit must be positive")
        return v

    @validator("offset")
    def validate_offset(cls, v):
        if v < 0:
            raise ValueError("offset must be non-negative")
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
    completed_at: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    config: Dict[str, Any] = Field(..., description="Job configuration")
    progress: Optional[float] = Field(None, description="Job progress (0-100)")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message if job failed")
    result_path: Optional[str] = Field(None, description="Path to job result file")

    model_config = ConfigDict(from_attributes=True)  # Replaces deprecated orm_mode=True


class LLMProcessResult(BaseModel):
    """Model for LLM processing result."""

    cluster_id: int = Field(..., description="Opinion cluster identifier")
    citation_analysis: CitationAnalysis = Field(
        ..., description="Citation analysis result"
    )


class ResolutionResult(BaseModel):
    """Model for citation resolution result."""

    cluster_id: int = Field(..., description="Opinion cluster identifier")
    resolved_citations: CombinedResolvedCitationAnalysis = Field(
        ..., description="Resolved citation analysis"
    )

    @classmethod
    def from_llm_result(cls, llm_result: LLMProcessResult):
        """Create ResolutionResult from LLMProcessResult."""

        # Use the built-in method from CombinedResolvedCitationAnalysis
        resolved = CombinedResolvedCitationAnalysis.from_citations(
            [llm_result.citation_analysis], llm_result.cluster_id
        )

        return cls(cluster_id=llm_result.cluster_id, resolved_citations=resolved)


class Neo4jLoadResult(BaseModel):
    """Model for Neo4j loading result."""

    cluster_ids: List[int] = Field(..., description="Loaded opinion cluster IDs")
    citation_count: int = Field(..., description="Number of citations loaded")

    @classmethod
    def from_loader_result(cls, loader_result: Dict[str, Any]):
        """Create Neo4jLoadResult from loader result."""
        return cls(
            cluster_ids=loader_result.get("cluster_ids", []),
            citation_count=loader_result.get("citation_count", 0),
        )


class PipelineResult(BaseModel):
    """Model for pipeline job result."""

    job_id: int = Field(..., description="Job identifier")
    job_type: JobType = Field(..., description="Job type")
    status: JobStatus = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    result: Dict[str, Any] = Field(..., description="Job result data")

    model_config = ConfigDict(from_attributes=True)  # Replaces deprecated orm_mode=True


class PipelineStats(BaseModel):
    """Model for pipeline statistics."""

    total_jobs: int = Field(..., description="Total number of jobs")
    jobs_by_type: Dict[str, int] = Field(..., description="Jobs by type")
    jobs_by_status: Dict[str, int] = Field(..., description="Jobs by status")
    avg_processing_time: Dict[str, float] = Field(
        ..., description="Average processing time by job type (seconds)"
    )
    recent_jobs: List[PipelineStatus] = Field(..., description="Recent jobs")
