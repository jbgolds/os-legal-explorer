from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    BackgroundTasks,
    File,
    UploadFile,
    Query,
)
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import os
from datetime import datetime

from ...database import get_db, get_neo4j
from .pipeline_model import (
    PipelineStatus,
    PipelineJob,
    PipelineResult,
    ExtractionConfig,
)
from . import pipeline_service

router = APIRouter(
    prefix="/api/pipeline",
    tags=["pipeline"],
    responses={404: {"description": "Not found"}},
)

# Simple API key security for localhost-only endpoints
API_KEY = os.getenv("PIPELINE_API_KEY", "local_development_key")
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


@router.post(
    "/extract", response_model=PipelineJob, dependencies=[Depends(verify_api_key)]
)
async def extract_opinions(
    config: ExtractionConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Extract opinions from PostgreSQL based on configuration.
    This is a long-running task that runs in the background.

    Args:
        config: Extraction configuration

    Returns:
        Job ID for tracking the extraction process
    """
    job_id = pipeline_service.create_job(db, "extract", config.dict())

    background_tasks.add_task(pipeline_service.run_extraction_job, db, job_id, config)

    return {"job_id": job_id, "status": "started"}


@router.post(
    "/process-llm", response_model=PipelineJob, dependencies=[Depends(verify_api_key)]
)
async def process_opinions_with_llm(
    job_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """
    Process extracted opinions through the LLM for citation analysis.
    This is a long-running task that runs in the background.

    Args:
        job_id: ID of the extraction job to process

    Returns:
        Job ID for tracking the LLM processing
    """
    # Verify the extraction job exists and is complete
    extraction_job = pipeline_service.get_job(db, job_id)
    if not extraction_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if extraction_job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Extraction job {job_id} is not completed (status: {extraction_job.status})",
        )

    llm_job_id = pipeline_service.create_job(
        db, "llm_process", {"extraction_job_id": job_id}
    )

    background_tasks.add_task(pipeline_service.run_llm_job, db, llm_job_id, job_id)

    return {"job_id": llm_job_id, "status": "started"}


@router.post(
    "/resolve-citations",
    response_model=PipelineJob,
    dependencies=[Depends(verify_api_key)],
)
async def resolve_citations(
    job_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """
    Resolve citations from LLM output to opinion cluster IDs.
    This is a long-running task that runs in the background.

    Args:
        job_id: ID of the LLM processing job

    Returns:
        Job ID for tracking the citation resolution
    """
    # Verify the LLM job exists and is complete
    llm_job = pipeline_service.get_job(db, job_id)
    if not llm_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if llm_job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"LLM job {job_id} is not completed (status: {llm_job.status})",
        )

    resolution_job_id = pipeline_service.create_job(
        db, "citation_resolution", {"llm_job_id": job_id}
    )

    background_tasks.add_task(
        pipeline_service.run_resolution_job, db, resolution_job_id, job_id
    )

    return {"job_id": resolution_job_id, "status": "started"}


@router.post(
    "/load-neo4j", response_model=PipelineJob, dependencies=[Depends(verify_api_key)]
)
async def load_neo4j(
    job_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    neo4j_session=Depends(get_neo4j),
):
    """
    Load resolved citations into Neo4j.
    This is a long-running task that runs in the background.

    Args:
        job_id: ID of the citation resolution job

    Returns:
        Job ID for tracking the Neo4j loading
    """
    # Verify the resolution job exists and is complete
    resolution_job = pipeline_service.get_job(db, job_id)
    if not resolution_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if resolution_job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Resolution job {job_id} is not completed (status: {resolution_job.status})",
        )

    neo4j_job_id = pipeline_service.create_job(
        db, "neo4j_load", {"resolution_job_id": job_id}
    )

    background_tasks.add_task(
        pipeline_service.run_neo4j_job, db, neo4j_session, neo4j_job_id, job_id
    )

    return {"job_id": neo4j_job_id, "status": "started"}


@router.post(
    "/upload-csv", response_model=PipelineJob, dependencies=[Depends(verify_api_key)]
)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a CSV file with opinions for processing.
    This is an alternative to the extract endpoint.

    Args:
        file: CSV file with opinions

    Returns:
        Job ID for tracking the upload process
    """
    # Create a temporary file to store the uploaded CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}_{file.filename}"
    file_path = os.path.join("/tmp", filename)

    # Save the uploaded file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create a job for the upload
    job_id = pipeline_service.create_job(
        db, "csv_upload", {"file_path": file_path, "original_filename": file.filename}
    )

    # Process the uploaded file in the background
    background_tasks.add_task(
        pipeline_service.process_uploaded_csv, db, job_id, file_path
    )

    return {"job_id": job_id, "status": "started"}


@router.get(
    "/job/{job_id}",
    response_model=PipelineStatus,
    dependencies=[Depends(verify_api_key)],
)
async def get_job_status(job_id: int, db: Session = Depends(get_db)):
    """
    Get the status of a pipeline job.

    Args:
        job_id: Job ID

    Returns:
        Job status information
    """
    job = pipeline_service.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return job


@router.get(
    "/jobs", response_model=List[PipelineStatus], dependencies=[Depends(verify_api_key)]
)
async def get_jobs(
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    Get a list of pipeline jobs with optional filtering.

    Args:
        job_type: Filter by job type
        status: Filter by job status
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip

    Returns:
        List of job status information
    """
    return pipeline_service.get_jobs(
        db, job_type=job_type, status=status, limit=limit, offset=offset
    )


@router.post(
    "/run-full-pipeline",
    response_model=List[PipelineJob],
    dependencies=[Depends(verify_api_key)],
)
async def run_full_pipeline(
    config: ExtractionConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    neo4j_session=Depends(get_neo4j),
):
    """
    Run the full pipeline from extraction to Neo4j loading.
    This is a convenience endpoint that chains all the steps.

    Args:
        config: Extraction configuration

    Returns:
        List of job IDs for each step in the pipeline
    """
    # Create jobs for each step
    extraction_job_id = pipeline_service.create_job(db, "extract", config.dict())
    llm_job_id = pipeline_service.create_job(
        db, "llm_process", {"extraction_job_id": extraction_job_id}
    )
    resolution_job_id = pipeline_service.create_job(
        db, "citation_resolution", {"llm_job_id": llm_job_id}
    )
    neo4j_job_id = pipeline_service.create_job(
        db, "neo4j_load", {"resolution_job_id": resolution_job_id}
    )

    # Run the full pipeline in the background
    background_tasks.add_task(
        pipeline_service.run_full_pipeline,
        db,
        neo4j_session,
        extraction_job_id,
        llm_job_id,
        resolution_job_id,
        neo4j_job_id,
        config,
    )

    return [
        {"job_id": extraction_job_id, "status": "queued"},
        {"job_id": llm_job_id, "status": "queued"},
        {"job_id": resolution_job_id, "status": "queued"},
        {"job_id": neo4j_job_id, "status": "queued"},
    ]
