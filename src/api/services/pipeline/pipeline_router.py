import os
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from src.api.database import get_db, get_neo4j
from src.api.services.pipeline import pipeline_service
from src.api.services.pipeline.pipeline_model import (ExtractionConfig,
                                                      PipelineJob,
                                                      PipelineStatus)
from src.api.services.pipeline.pipeline_single_cluster import \
    process_single_cluster
from starlette.concurrency import run_in_threadpool

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
    job_id = pipeline_service.create_job("extract", config.model_dump())

    background_tasks.add_task(pipeline_service.run_extraction_job, job_id, config)

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
    extraction_job = pipeline_service.get_job(job_id)
    if not extraction_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if extraction_job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Extraction job {job_id} is not completed (status: {extraction_job['status']})",
        )

    llm_job_id = pipeline_service.create_job(
        "llm_process", {"extraction_job_id": job_id}
    )

    background_tasks.add_task(pipeline_service.run_llm_job, llm_job_id, job_id)

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
    llm_job = pipeline_service.get_job(job_id)
    if not llm_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if llm_job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"LLM job {job_id} is not completed (status: {llm_job['status']})",
        )

    resolution_job_id = pipeline_service.create_job(
        "citation_resolution", {"llm_job_id": job_id}
    )

    background_tasks.add_task(
        pipeline_service.run_resolution_job, resolution_job_id, job_id
    )

    return {"job_id": resolution_job_id, "status": "started"}


@router.post(
    "/load-neo4j", response_model=PipelineJob, dependencies=[Depends(verify_api_key)]
)
async def load_neo4j(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    # neo4j_session=Depends(get_neo4j),
    file_path: Optional[str] = None,
    job_id: Optional[int] = None,
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
    if not file_path and not job_id:
        raise HTTPException(status_code=400, detail="Either file_path or job_id must be provided")
    
    
    if job_id:
        resolution_job = pipeline_service.get_job( job_id)      
        if not resolution_job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        


        if resolution_job["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Resolution job {job_id} is not completed (status: {resolution_job['status']})",
            )

    neo4j_job_id = pipeline_service.create_job(
        "neo4j_load", {"resolution_job_id": job_id}
    )

    background_tasks.add_task(
        pipeline_service.run_neo4j_job,     neo4j_job_id, job_id, file_path
    )

    return {"job_id": neo4j_job_id, "status": "started"}



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
    job = pipeline_service.get_job( job_id)
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
        job_type=job_type, status=status, limit=limit, offset=offset
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
    # neo4j_session=Depends(get_neo4j),
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
    extraction_job_id = pipeline_service.create_job( "extract", config.model_dump())
    llm_job_id = pipeline_service.create_job(
         "llm_process", {"extraction_job_id": extraction_job_id} 
    )
    resolution_job_id = pipeline_service.create_job(
         "citation_resolution", {"llm_job_id": llm_job_id}
    )
    neo4j_job_id = pipeline_service.create_job(
         "neo4j_load", {"resolution_job_id": resolution_job_id}
    )

    # Run the full pipeline in the background
    result = await run_in_threadpool(
        pipeline_service.run_full_pipeline,
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


@router.post(
    "/process-cluster/{cluster_id}",
    response_model=dict,
)
async def process_single_cluster_endpoint(
    cluster_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Process a single cluster through the full pipeline.

    This endpoint will extract the cluster, process it with LLM,
    resolve citations, and load them into Neo4j.

    This endpoint is publicly accessible without an API key.

    Args:
        cluster_id: ID of the cluster to process

    Returns:
        List of job IDs for each step in the pipeline
    """
    # Create a custom extraction config for this cluster
    result = await run_in_threadpool(
        process_single_cluster,
        cluster_id,
    )
    # Create jobs for each step

    # return HTTP status code accepted
    return {"message": "Cluster processing started"}
