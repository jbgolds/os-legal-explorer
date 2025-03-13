import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import (APIRouter, BackgroundTasks, Depends, File, HTTPException,
                     Query, UploadFile)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from google.genai.types import BatchJob
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import pipeline service components for data processing
from src.api.database import get_db, get_neo4j
from src.api.services.pipeline import pipeline_service
from src.api.services.pipeline.pipeline_model import (
    CombinedResolvedCitationAnalysis, ExtractionConfig)
from src.api.services.pipeline.pipeline_service import (
    create_combined_analyses, handle_empty_citations,
    serialize_and_save_citations)
# Import the BatchGeminiClient
from src.llm_extraction.batch_gemini import BatchGeminiClient
from src.llm_extraction.models import CitationAnalysis
from src.neo4j_db.neomodel_loader import neomodel_loader

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for development
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)

# Create logger for this module
logger = logging.getLogger("batch_gemini_router")

# Create router
router = APIRouter(
    prefix="/api/batch-gemini",
    tags=["batch-gemini"],
    responses={404: {"description": "Not found"}},
)

# Set up API key security
API_KEY = os.getenv("BATCH_GEMINI_API_KEY", "local_development_key")
api_key_header = APIKeyHeader(name="X-API-Key")

# Define job status constants
class JobStatus:
    """Constants for job status values."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    SUBMITTED = "SUBMITTED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CLEANING = "CLEANING"
    RESOLVING = "RESOLVING"
    LOADING_NEO4J = "LOADING_NEO4J"
    FAILED = "FAILED"

# Define Pydantic models for request/response
class BatchGeminiJob(BaseModel):
    """Model representing a batch Gemini job."""
    job_id: str
    status: str
    created_at: str
    file_path: Optional[str] = None
    row_count: Optional[int] = None
    batch_job_name: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    # Fields for tracking pipeline progress
    results_file: Optional[str] = None
    cleaned_file: Optional[str] = None
    resolved_file: Optional[str] = None
    neo4j_loaded: Optional[bool] = None
    
class BatchGeminiStatus(BaseModel):
    """Model representing the status of a batch Gemini job."""
    job_id: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    # Pipeline stage tracking
    gemini_complete: Optional[bool] = None
    cleaning_complete: Optional[bool] = None
    resolution_complete: Optional[bool] = None
    neo4j_complete: Optional[bool] = None
    
class BatchGeminiConfig(BaseModel):
    """Configuration for batch Gemini processing."""
    # text_column: str = "text"  # Removed as not needed
    # Add options for pipeline steps
    run_cleaning: bool = True
    run_resolution: bool = True
    run_neo4j_load: bool = True

class ContinuePollRequest(BaseModel):
    """Request model for continuing polling of a Gemini batch job."""
    batch_job_name: str
    original_job_id: Optional[str] = None

class ContinuePipelineRequest(BaseModel):
    """Request model for continuing a pipeline that was interrupted."""
    batch_job_name: str
    original_job_id: Optional[str] = None
    # Pipeline configuration
    run_cleaning: bool = True
    run_resolution: bool = True
    run_neo4j_load: bool = True

# Mapping from Gemini job states to our status values
GEMINI_STATE_MAPPING = {
    "JOB_STATE_SUCCEEDED": JobStatus.COMPLETED,
    "JOB_STATE_FAILED": JobStatus.FAILED,
    "JOB_STATE_RUNNING": JobStatus.PROCESSING,
    "JOB_STATE_PENDING": JobStatus.QUEUED,
    "JOB_STATE_CANCELLING": JobStatus.PROCESSING,
    "JOB_STATE_CANCELLED": JobStatus.FAILED,
}

# In-memory job storage (replace with a database in production)
job_store: Dict[str, Dict[str, Any]] = {}

def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify that the API key is valid."""
    logger.debug(f"Verifying API key: {'valid' if api_key == API_KEY else 'invalid'}")
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempted: {api_key[:5]}...")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

def save_to_tmp(data: Any, filename: str, as_json: bool = False) -> str:
    """
    Save data to a temporary file.
    
    Args:
        data: Data to save (DataFrame or list/dict for JSON)
        filename: Name for the temporary file
        as_json: Whether to save as JSON (True) or CSV (False)
        
    Returns:
        Path to the saved file
    """
    logger.debug(f"Saving data to temporary file: {filename} (as_json={as_json})")
    tmp_path = os.path.join("/tmp", filename)
    try:
        if as_json:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            # Assume pandas DataFrame for non-JSON data
            if isinstance(data, pd.DataFrame):
                data.to_csv(tmp_path, index=False)
            else:
                raise ValueError(f"Expected DataFrame for CSV output, got {type(data)}")
                
        logger.debug(f"Successfully saved data to {tmp_path}")
        return tmp_path
    except Exception as e:
        logger.exception(f"Error saving data to {tmp_path}: {str(e)}")
        raise

def process_batch_job(
    client: BatchGeminiClient, 
    df: pd.DataFrame, 
    job_id: str
):
    """
    Process a batch job with Gemini.
    
    Args:
        client: The BatchGeminiClient instance.
        df: DataFrame containing the data to process.
        job_id: The unique identifier for this job.
        
    Returns:
        The job object from the Gemini API.
    """
    try:
        # Convert DataFrame to JSONL
        jsonl_file_name = f"/tmp/batch_{job_id}.jsonl"
        
        # Convert DataFrame to JSONL
        with open(jsonl_file_name, 'w') as f:
            for _, row in df.iterrows():
                json.dump(row.to_dict(), f)
                f.write('\n')
        
        logger.info(f"[JOB:{job_id}] Converted DataFrame to JSONL: {jsonl_file_name}")
        
        # Submit the batch job to Gemini
        # Using fixed "text" column instead of configurable text_column
        batch_job = client.submit_batch_job(df, jsonl_file_name, text_column="text")
        
        logger.info(f"[JOB:{job_id}] Submitted batch job to Gemini: {batch_job.name}")
        
        return batch_job
    except Exception as e:
        logger.exception(f"[JOB:{job_id}] Error processing batch job: {str(e)}")
        raise

def check_job_status_background(client: BatchGeminiClient, job_id: str):
    """
    Background task to check the status of a batch job.
    
    Args:
        client: An instance of BatchGeminiClient.
        job_id: The unique identifier of the job.
    """
    logger.info(f"[JOB:{job_id}] Starting status check")
    try:
        if job_id not in job_store:
            logger.error(f"[JOB:{job_id}] Job not found in job store")
            return
            
        job_data = job_store[job_id]
        if "batch_job_name" not in job_data:
            logger.error(f"[JOB:{job_id}] No batch_job_name in job data")
            return
        
        batch_job_name = job_data["batch_job_name"]
        logger.debug(f"[JOB:{job_id}] Checking status for batch job: {batch_job_name}")
        
        # Create a BatchJob object with the name
        batch_job = BatchJob(name=batch_job_name)
        
        # Check the status of the batch job
        logger.debug(f"[JOB:{job_id}] Making API call to check status")
        updated_job = client.check_batch_job_status(batch_job)
        
        # Update the job store with the current status
        logger.info(f"[JOB:{job_id}] Status from API: {updated_job.state}")
        job_store[job_id].update({
            "status": updated_job.state,
            "updated_at": pd.Timestamp.now().isoformat()
        })
        
        logger.debug(f"[JOB:{job_id}] Updated job store with new status")
    except Exception as e:
        error_message = f"Error checking batch job status: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": "ERROR",
            "error": error_message
        })

def retrieve_results_background(client: BatchGeminiClient, job_id: str):
    """
    Background task to retrieve results from a completed batch job.
    
    Args:
        client: An instance of BatchGeminiClient.
        job_id: The unique identifier of the job.
    """
    logger.info(f"[JOB:{job_id}] Starting results retrieval")
    try:
        if job_id not in job_store:
            logger.error(f"[JOB:{job_id}] Job not found in job store")
            return
            
        job_data = job_store[job_id]
        if "batch_job_name" not in job_data:
            logger.error(f"[JOB:{job_id}] No batch_job_name in job data")
            return
        
        batch_job_name = job_data["batch_job_name"]    
        logger.debug(f"[JOB:{job_id}] Retrieving results for batch job: {batch_job_name}")
        
        # Create a BatchJob object with the name
        batch_job = BatchJob(name=batch_job_name)
        
        # Check the status of the batch job to make sure it's completed
        logger.debug(f"[JOB:{job_id}] Checking status before retrieving results")
        updated_job = client.check_batch_job_status(batch_job)
        
        logger.info(f"[JOB:{job_id}] Current job state: {updated_job.state}")
        if updated_job.state == "JOB_STATE_SUCCEEDED":
            # Retrieve the results from the batch job
            logger.info(f"[JOB:{job_id}] Job succeeded, retrieving results")
            results_df = client.retrieve_batch_job_results(updated_job)
            
            if results_df is not None:
                logger.debug(f"[JOB:{job_id}] Results retrieved with {len(results_df)} rows")
                # Save results to a temporary CSV file
                results_file = save_to_tmp(results_df, f"results_{job_id}.csv")
                
                # Update job status with the results file path
                logger.info(f"[JOB:{job_id}] Results saved to {results_file}")
                job_store[job_id].update({
                    "status": "COMPLETED",
                    "results_file": results_file,
                    "message": "Results retrieved successfully",
                    "gemini_complete": True
                })
            else:
                logger.error(f"[JOB:{job_id}] Results retrieval returned None")
                job_store[job_id].update({
                    "status": "ERROR",
                    "error": "Results retrieval returned None"
                })
        else:
            logger.warning(f"[JOB:{job_id}] Job not ready for results retrieval. State: {updated_job.state}")
            job_store[job_id].update({
                "status": updated_job.state,
                "message": f"Job is not completed yet. Current state: {updated_job.state}"
            })
    except Exception as e:
        error_message = f"Error retrieving batch job results: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": "ERROR",
            "error": error_message
        })

def clean_data_background(job_id: str, db: Session):
    """
    Background task to clean the data retrieved from a batch job.
    
    Args:
        job_id: The unique identifier of the job.
        db: Database session.
    """
    logger.info(f"[JOB:{job_id}] Starting data cleaning")
    try:
        if job_id not in job_store:
            logger.error(f"[JOB:{job_id}] Job not found in job store")
            return
            
        job_data = job_store[job_id]
        
        # Check if Gemini processing is complete and we have results
        if job_data.get("status") != JobStatus.COMPLETED or "results_file" not in job_data:
            logger.error(f"[JOB:{job_id}] Job not completed or no results file available")
            job_store[job_id].update({
                "status": JobStatus.ERROR,
                "error": "Cannot clean data: job not completed or no results file"
            })
            return
        
        # Update status to cleaning
        job_store[job_id].update({
            "status": JobStatus.CLEANING,
            "message": "Cleaning data"
        })
        
        # Read the results file
        results_file = job_data["results_file"]
        logger.debug(f"[JOB:{job_id}] Reading results from {results_file}")
        df = pd.read_csv(results_file)
        
        # Clean the data using the pipeline service function
        logger.info(f"[JOB:{job_id}] Cleaning data with {len(df)} rows")
        cleaned_df = pipeline_service.clean_extracted_opinions(df)
        logger.info(f"[JOB:{job_id}] Used clean_extracted_opinions")
        
        # Save the cleaned data to a file
        cleaned_file = save_to_tmp(cleaned_df, f"cleaned_{job_id}.csv", as_json=False)
        
        # Update job status
        logger.info(f"[JOB:{job_id}] Data cleaning completed, saved to {cleaned_file}")
        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "cleaned_file": cleaned_file,
            "message": "Data cleaning completed",
            "cleaning_complete": True
        })
    except Exception as e:
        error_message = f"Error cleaning data: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })

def resolve_citations_background(job_id: str, db: Session):
    """
    Background task to resolve citations in the cleaned data.
    
    Args:
        job_id: The unique identifier of the job.
        db: Database session.
    """
    logger.info(f"[JOB:{job_id}] Starting citation resolution")
    try:
        if job_id not in job_store:
            logger.error(f"[JOB:{job_id}] Job not found in job store")
            return
            
        job_data = job_store[job_id]
        
        # Check if cleaning is complete
        if not job_data.get("cleaning_complete") or "cleaned_file" not in job_data:
            logger.error(f"[JOB:{job_id}] Data cleaning not completed or no cleaned file available")
            job_store[job_id].update({
                "status": JobStatus.ERROR,
                "error": "Cannot resolve citations: data cleaning not completed or no cleaned file"
            })
            return
        
        # Update status to resolving
        job_store[job_id].update({
            "status": JobStatus.RESOLVING,
            "message": "Resolving citations"
        })
        
        # Read the cleaned file
        cleaned_file = job_data["cleaned_file"]
        logger.debug(f"[JOB:{job_id}] Reading cleaned data from {cleaned_file}")
        df = pd.read_csv(cleaned_file)
        
        # Convert DataFrame to list of CitationAnalysis objects
        logger.debug(f"[JOB:{job_id}] Converting DataFrame to CitationAnalysis objects")
        citation_analyses = []
        for _, row in df.iterrows():
            try:
                # Create a CitationAnalysis object from the row data
                analysis = CitationAnalysis(
                    date=row.get('date', '2023-01-01'),  # Default date if not available
                    brief_summary=row.get('ai_summary', ''),
                    majority_opinion_citations=[],  # These would need proper conversion from JSON
                    concurring_opinion_citations=[],  # These would need proper conversion from JSON
                    dissenting_citations=[]  # These would need proper conversion from JSON
                )
                citation_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"[JOB:{job_id}] Error converting row to CitationAnalysis: {str(e)}")
                # Continue with the next row
        
        # Resolve citations
        logger.info(f"[JOB:{job_id}] Resolving citations for {len(citation_analyses)} analyses")
        resolved_analyses = []
        for analysis in citation_analyses:
            try:
                # This is a simplified version - in reality, you'd use the citation resolution logic
                # from pipeline_service.py
                resolved_analysis = CombinedResolvedCitationAnalysis(
                    date=analysis.date if hasattr(analysis, "date") else "2023-01-01",  # Default date if not available
                    cluster_id=int(analysis.cluster_id) if hasattr(analysis, "cluster_id") else 0,
                    brief_summary=analysis.ai_summary if hasattr(analysis, "ai_summary") else "No summary available",
                    # Convert citations to the appropriate format if needed
                    majority_opinion_citations=[],  # Placeholder - would need proper conversion
                    concurring_opinion_citations=[],  # Placeholder
                    dissenting_citations=[]  # Placeholder
                )
                resolved_analyses.append(resolved_analysis)
            except Exception as e:
                logger.warning(f"[JOB:{job_id}] Error resolving citations for analysis {analysis.cluster_id}: {str(e)}")
                # Continue with the next analysis
        
        # Save the resolved analyses to a file
        resolved_file = save_to_tmp(
            [analysis.model_dump() for analysis in resolved_analyses],
            f"resolved_{job_id}",
            as_json=True
        )
        
        # Update job status
        logger.info(f"[JOB:{job_id}] Citation resolution completed, saved to {resolved_file}")
        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "resolved_file": resolved_file,
            "message": "Citation resolution completed",
            "resolution_complete": True
        })
    except Exception as e:
        error_message = f"Error resolving citations: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })

def load_neo4j_background(job_id: str, db: Session, neo4j_session):
    """
    Background task to load resolved citations into Neo4j.
    
    Args:
        job_id: The unique identifier of the job.
        db: Database session.
        neo4j_session: Neo4j database session.
    """
    logger.info(f"[JOB:{job_id}] Starting Neo4j loading")
    try:
        if job_id not in job_store:
            logger.error(f"[JOB:{job_id}] Job not found in job store")
            return
            
        job_data = job_store[job_id]
        
        # Check if resolution is complete
        if not job_data.get("resolution_complete") or "resolved_file" not in job_data:
            logger.error(f"[JOB:{job_id}] Citation resolution not completed or no resolved file available")
            job_store[job_id].update({
                "status": JobStatus.ERROR,
                "error": "Cannot load to Neo4j: citation resolution not completed or no resolved file"
            })
            return
        
        # Update status to loading Neo4j
        job_store[job_id].update({
            "status": JobStatus.LOADING_NEO4J,
            "message": "Loading data into Neo4j"
        })
        
        # Read the resolved file
        resolved_file = job_data["resolved_file"]
        logger.debug(f"[JOB:{job_id}] Reading resolved data from {resolved_file}")
        with open(resolved_file, 'r') as f:
            resolved_data = json.load(f)
        
        # Create a NeomodelLoader instance
        loader = neomodel_loader
        
        # Load the resolved data into Neo4j
        logger.info(f"[JOB:{job_id}] Loading {len(resolved_data)} analyses into Neo4j")
        for analysis_dict in resolved_data:
            try:
                # Convert the dictionary to a CombinedResolvedCitationAnalysis object
                analysis = CombinedResolvedCitationAnalysis(**analysis_dict)
                
                # Load the analysis into Neo4j
                loader.load_enriched_citations([analysis], data_source="gemini_api")
            except Exception as e:
                logger.warning(f"[JOB:{job_id}] Error loading analysis into Neo4j: {str(e)}")
                # Continue with the next analysis
        
        # Update job status
        logger.info(f"[JOB:{job_id}] Neo4j loading completed")
        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "neo4j_loaded": True,
            "message": "Neo4j loading completed",
            "neo4j_complete": True
        })
    except Exception as e:
        error_message = f"Error loading to Neo4j: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })

@router.post(
    "/submit", 
    response_model=BatchGeminiJob, 
    dependencies=[Depends(verify_api_key)]
)
async def submit_batch_job_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Submit a batch job for Gemini processing.
    
    Args:
        background_tasks: FastAPI background tasks.
        file: CSV file with input data.
        
    Returns:
        A JSON object with the assigned job ID and job status.
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"[JOB:{job_id}] New job submission request received for file: {file.filename}")
    
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("/tmp", f"upload_{job_id}.csv")
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        logger.debug(f"[JOB:{job_id}] Saved uploaded file to {temp_file_path}")
        
        # Read the CSV file
        try:
            df = pd.read_csv(temp_file_path)
            logger.debug(f"[JOB:{job_id}] Read CSV with {len(df)} rows and columns: {list(df.columns)}")
        except Exception as e:
            error_msg = f"Failed to read CSV file: {str(e)}"
            logger.error(f"[JOB:{job_id}] {error_msg}")
            return JSONResponse(
                status_code=400,
                content={"error": error_msg}
            )
        
        # Validate that the text column exists
        if "text" not in df.columns:
            error_msg = f"Text column 'text' not found. Available columns: {', '.join(df.columns)}"
            logger.error(f"[JOB:{job_id}] {error_msg}")
            return JSONResponse(
                status_code=400,
                content={"error": error_msg}
            )
        
        # Create a new job entry
        now = pd.Timestamp.now().isoformat()
        job_store[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "created_at": now,
            "file_path": temp_file_path,
            "row_count": len(df),
            "message": "Job queued for processing"
        }
        
        logger.info(f"[JOB:{job_id}] Created job entry with {len(df)} rows")
        
        # Start processing in the background
        gemini_client = BatchGeminiClient()
        background_tasks.add_task(
            process_batch_job,
            gemini_client, 
            df, 
            job_id
        )
        
        return job_store[job_id]
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

@router.get(
    "/status/{job_id}", 
    response_model=BatchGeminiStatus, 
    dependencies=[Depends(verify_api_key)]
)
async def check_batch_job_status_endpoint(
    background_tasks: BackgroundTasks,
    job_id: str,
    wait: bool = Query(False, description="Whether to wait for job completion (long polling)")
):
    """
    Check the status of a submitted batch job.
    
    Args:
        background_tasks: FastAPI background tasks.
        job_id: The unique identifier of the batch job.
        wait: If True, the request will wait until the job completes or fails (long polling).
        
    Returns:
        A JSON object containing the job status.
    """
    logger.info(f"[JOB:{job_id}] Status check request received, wait={wait}")
    
    if job_id not in job_store:
        logger.warning(f"[JOB:{job_id}] Job not found in job store")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Always check the latest status if we have a batch job name
    if "batch_job_name" in job_store[job_id]:
        logger.info(f"[JOB:{job_id}] Checking latest status")
        gemini_client = BatchGeminiClient()
        
        if wait:
            # For long polling, we'll check the status directly and wait for completion
            try:
                batch_job_name = job_store[job_id]["batch_job_name"]
                batch_job = BatchJob(name=batch_job_name)
                
                # Initial status check
                updated_job = gemini_client.check_batch_job_status(batch_job)
                current_status = updated_job.state
                
                # Update job store with current status
                job_store[job_id].update({
                    "status": current_status,
                    "updated_at": pd.Timestamp.now().isoformat()
                })
                
                # Define terminal states
                terminal_states = [JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.FAILED]
                
                # Poll until job reaches a terminal state or timeout (30 minutes max)
                start_time = time.time()
                timeout = 30 * 60  # 30 minutes in seconds
                
                while current_status not in terminal_states and (time.time() - start_time) < timeout:
                    # Wait before checking again
                    await asyncio.sleep(10)  # Check every 10 seconds
                    
                    # Check status again
                    updated_job = gemini_client.check_batch_job_status(batch_job)
                    current_status = updated_job.state
                    
                    # Update job store
                    job_store[job_id].update({
                        "status": current_status,
                        "updated_at": pd.Timestamp.now().isoformat()
                    })
                    
                    logger.debug(f"[JOB:{job_id}] Long polling update: {current_status}")
                    
                    # If we've reached a terminal state, break out of the loop
                    if current_status in terminal_states:
                        logger.info(f"[JOB:{job_id}] Job reached terminal state: {current_status}")
                        
                        # If completed, trigger the results retrieval
                        if current_status == JobStatus.COMPLETED:
                            background_tasks.add_task(retrieve_results_background, gemini_client, job_id)
                        
                        break
                
                # If we timed out
                if (time.time() - start_time) >= timeout:
                    logger.warning(f"[JOB:{job_id}] Long polling timed out after {timeout} seconds")
                    job_store[job_id].update({
                        "message": f"Long polling timed out after {timeout} seconds"
                    })
                
                return job_store[job_id]
                
            except Exception as e:
                error_message = f"Error during long polling: {str(e)}"
                logger.exception(f"[JOB:{job_id}] {error_message}")
                job_store[job_id].update({
                    "status": JobStatus.ERROR,
                    "error": error_message
                })
                return job_store[job_id]
        else:
            # For regular polling, use the background task as before
            background_tasks.add_task(check_job_status_background, gemini_client, job_id)
            return {**job_store[job_id], "message": "Status check has been queued"}
    
    # Return the current status from the job store
    logger.debug(f"[JOB:{job_id}] Returning current status: {job_store[job_id].get('status')}")
    return job_store[job_id]

@router.get(
    "/results/{job_id}", 
    dependencies=[Depends(verify_api_key)]
)
async def retrieve_batch_job_results_endpoint(
    background_tasks: BackgroundTasks,
    job_id: str
):
    """
    Retrieve the results for a completed batch job.
    
    Args:
        background_tasks: FastAPI background tasks.
        job_id: The unique identifier of the batch job.
        
    Returns:
        A JSON object with the job results or status.
    """
    logger.info(f"[JOB:{job_id}] Results retrieval request received")
    
    if job_id not in job_store:
        logger.warning(f"[JOB:{job_id}] Job not found in job store")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_data = job_store[job_id]
    logger.debug(f"[JOB:{job_id}] Current job status: {job_data.get('status')}")
    
    # If job is not completed, check if we can retrieve results
    if job_data.get("status") != "COMPLETED":
        if "batch_job_name" not in job_data:
            logger.error(f"[JOB:{job_id}] No batch_job_name in job data")
            raise HTTPException(
                status_code=400, 
                detail="Job has not been submitted to Gemini yet"
            )
        
        logger.info(f"[JOB:{job_id}] Adding results retrieval to background tasks")
        gemini_client = BatchGeminiClient()
        background_tasks.add_task(retrieve_results_background, gemini_client, job_id)
        
        return {
            **job_data,
            "message": "Results retrieval has been queued"
        }
    
    # If the job is completed and we have results
    if "results_file" in job_data:
        logger.info(f"[JOB:{job_id}] Job completed, returning results from {job_data['results_file']}")
        try:
            # Read the results from the file
            logger.debug(f"[JOB:{job_id}] Reading results file")
            results_df = pd.read_csv(job_data["results_file"])
            
            # Return only the first 100 rows to avoid overwhelming the response
            logger.info(f"[JOB:{job_id}] Returning preview of {min(100, len(results_df))} rows out of {len(results_df)}")
            return {
                **job_data,
                "results_preview": results_df.head(100).to_dict(orient="records"),
                "total_rows": len(results_df)
            }
        except Exception as e:
            logger.exception(f"[JOB:{job_id}] Error reading results file: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error reading results file: {str(e)}"
            )
    
    # If job is completed but no results file is found
    logger.warning(f"[JOB:{job_id}] Job is completed but no results file was found")
    return {
        **job_data,
        "message": "Job is completed but no results file was found"
    }

@router.get(
    "/jobs", 
    dependencies=[Depends(verify_api_key)]
)
async def list_jobs(
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    List all batch jobs, optionally filtered by status.
    
    Args:
        status: Filter jobs by this status.
        limit: Maximum number of jobs to return.
        offset: Number of jobs to skip.
        
    Returns:
        A list of jobs matching the criteria.
    """
    logger.info(f"List jobs request received, status={status}, limit={limit}, offset={offset}")
    
    # Filter jobs by status if provided
    filtered_jobs = list(job_store.values())
    if status:
        logger.debug(f"Filtering jobs by status: {status}")
        filtered_jobs = [job for job in filtered_jobs if job.get("status") == status]
    
    # Sort by creation time (newest first)
    filtered_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    # Apply pagination
    paginated_jobs = filtered_jobs[offset:offset + limit]
    
    logger.info(f"Returning {len(paginated_jobs)} jobs (from {len(filtered_jobs)} total)")
    return {
        "jobs": paginated_jobs,
        "total": len(filtered_jobs),
        "limit": limit,
        "offset": offset
    }

@router.delete(
    "/job/{job_id}", 
    dependencies=[Depends(verify_api_key)]
)
async def delete_job(
    job_id: str
):
    """
    Delete a job from the job store.
    
    Args:
        job_id: The unique identifier of the job to delete.
        
    Returns:
        A confirmation message.
    """
    logger.info(f"[JOB:{job_id}] Delete job request received")
    
    if job_id not in job_store:
        logger.warning(f"[JOB:{job_id}] Job not found in job store")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Get the job data before deleting
    job_data = job_store[job_id]
    
    # Delete any temporary files associated with the job
    file_paths = [
        job_data.get("file_path"),
        job_data.get("results_file"),
        job_data.get("cleaned_file"),
        job_data.get("resolved_file")
    ]
    
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                logger.debug(f"[JOB:{job_id}] Removing file: {path}")
                os.remove(path)
            except Exception as e:
                logger.warning(f"[JOB:{job_id}] Error removing temporary file {path}: {str(e)}")
    
    # Remove the job from the store
    logger.info(f"[JOB:{job_id}] Removing job from store")
    del job_store[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

@router.post(
    "/clean/{job_id}",
    response_model=BatchGeminiStatus,
    dependencies=[Depends(verify_api_key)]
)
async def clean_data_endpoint(
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Clean the data retrieved from a batch Gemini job.
    
    Args:
        job_id: The unique identifier of the job.
        background_tasks: FastAPI background tasks.
        db: Database session.
        
    Returns:
        A JSON object with the job status.
    """
    logger.info(f"[JOB:{job_id}] Data cleaning request received")
    
    if job_id not in job_store:
        logger.warning(f"[JOB:{job_id}] Job not found in job store")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_data = job_store[job_id]
    
    # Check if Gemini processing is complete
    if not job_data.get("gemini_complete") or "results_file" not in job_data:
        logger.error(f"[JOB:{job_id}] Gemini processing not completed or no results file available")
        raise HTTPException(
            status_code=400,
            detail="Cannot clean data: Gemini processing not completed or no results file available"
        )
    
    # Add the data cleaning task to background tasks
    logger.info(f"[JOB:{job_id}] Adding data cleaning to background tasks")
    background_tasks.add_task(clean_data_background, job_id, db)
    
    # Update job status
    job_store[job_id].update({
        "message": "Data cleaning has been queued"
    })
    
    return job_store[job_id]

@router.post(
    "/resolve/{job_id}",
    response_model=BatchGeminiStatus,
    dependencies=[Depends(verify_api_key)]
)
async def resolve_citations_endpoint(
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Resolve citations in the cleaned data.
    
    Args:
        job_id: The unique identifier of the job.
        background_tasks: FastAPI background tasks.
        db: Database session.
        
    Returns:
        A JSON object with the job status.
    """
    logger.info(f"[JOB:{job_id}] Citation resolution request received")
    
    if job_id not in job_store:
        logger.warning(f"[JOB:{job_id}] Job not found in job store")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_data = job_store[job_id]
    
    # Check if data cleaning is complete
    if not job_data.get("cleaning_complete") or "cleaned_file" not in job_data:
        logger.error(f"[JOB:{job_id}] Data cleaning not completed or no cleaned file available")
        raise HTTPException(
            status_code=400,
            detail="Cannot resolve citations: data cleaning not completed or no cleaned file available"
        )
    
    # Add the citation resolution task to background tasks
    logger.info(f"[JOB:{job_id}] Adding citation resolution to background tasks")
    background_tasks.add_task(resolve_citations_background, job_id, db)
    
    # Update job status
    job_store[job_id].update({
        "message": "Citation resolution has been queued"
    })
    
    return job_store[job_id]

@router.post(
    "/load-neo4j/{job_id}",
    response_model=BatchGeminiStatus,
    dependencies=[Depends(verify_api_key)]
)
async def load_neo4j_endpoint(
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    neo4j_session = Depends(get_neo4j)
):
    """
    Load the resolved citations into Neo4j.
    
    Args:
        job_id: The unique identifier for the job
        background_tasks: FastAPI background tasks
        db: Database session
        neo4j_session: Neo4j database session
        
    Returns:
        Updated job status
    """
    logger.info(f"[JOB:{job_id}] Request to load data into Neo4j")
    
    # Check if the job exists
    if job_id not in job_store:
        return JSONResponse(
            status_code=404,
            content={"error": f"Job {job_id} not found"}
        )
    
    # Check if the job has resolved citations
    if not job_store[job_id].get("resolved_file"):
        return JSONResponse(
            status_code=400,
            content={"error": "Job does not have resolved citations. Run citation resolution first."}
        )
    
    # Update job status
    job_store[job_id].update({
        "status": JobStatus.LOADING_NEO4J,
        "message": "Loading data into Neo4j"
    })
    
    # Start the Neo4j loading in the background
    background_tasks.add_task(
        load_neo4j_background,
        job_id,
        db,
        neo4j_session
    )
    
    return job_store[job_id]

@router.post(
    "/run-full-pipeline",
    response_model=BatchGeminiJob,
    dependencies=[Depends(verify_api_key)],
)
async def run_full_pipeline(
    background_tasks: BackgroundTasks,
    config: ExtractionConfig,
    batch_config: BatchGeminiConfig = Depends(),
    db: Session = Depends(get_db),
    neo4j_session = Depends(get_neo4j)
):
    """
    Run the full pipeline from database extraction to Neo4j loading.
    
    This endpoint uses SQL query filters as the data source input, similar to
    submit-with-filters, but handles the full pipeline execution in one go.
    
    Args:
        background_tasks: FastAPI background tasks
        config: Extraction configuration with filters
        batch_config: Configuration options for the pipeline steps
        db: Database session
        neo4j_session: Neo4j database session
        
    Returns:
        Job information
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"[JOB:{job_id}] New full pipeline request received with filters: {config}")
    
    try:
        # Create a new job entry
        now = pd.Timestamp.now().isoformat()
        job_store[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "created_at": now,
            "config": config.model_dump(),
            "batch_config": batch_config.model_dump(),
            "message": "Full pipeline job queued for processing with database filters"
        }
        
        logger.info(f"[JOB:{job_id}] Created full pipeline job with filters")
        
        # Start the first step of the pipeline in the background
        background_tasks.add_task(
            start_pipeline_with_filters,
            job_id,
            config,
            batch_config,
            db,
            neo4j_session
        )
        
        return job_store[job_id]
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

async def start_pipeline_with_filters(
    job_id: str,
    config: ExtractionConfig,
    batch_config: BatchGeminiConfig,
    db: Session,
    neo4j_session
):
    """
    Start the pipeline by submitting a batch job with database filters.
    
    Args:
        job_id: The unique identifier for this job
        config: Extraction configuration with filters
        batch_config: Configuration options for the pipeline steps
        db: Database session
        neo4j_session: Neo4j database session
    """
    try:
        # Step 1: Process with Gemini using database filters
        logger.info(f"[JOB:{job_id}] Starting Gemini processing with database filters")
        job_store[job_id].update({
            "status": JobStatus.PROCESSING,
            "message": "Extracting opinions from database and processing with Gemini"
        })
        
        # Create the BatchGeminiClient
        gemini_client = BatchGeminiClient()
        
        # Generate a JSONL file name
        jsonl_file_name = f"batch_{job_id}.jsonl"
        
        # Extract opinions, clean them, and submit the batch job
        batch_job = gemini_client.submit_batch_job_from_config(
            config, 
            db.bind,  # Pass the database connection
            jsonl_file_name
        )
        
        # Update job status
        logger.info(f"[JOB:{job_id}] Submitted batch job to Gemini: {batch_job.name}")
        job_store[job_id].update({
            "status": JobStatus.SUBMITTED,
            "batch_job_name": batch_job.name,
            "message": f"Batch job submitted to Gemini: {batch_job.name}"
        })
        
        # Schedule a background task to check the job status
        asyncio.create_task(
            check_pipeline_job_status(
                job_id,
                batch_job,
                batch_config,
                db,
                neo4j_session
            )
        )
        
    except Exception as e:
        error_message = f"Error starting pipeline with filters: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })

async def check_pipeline_job_status(
    job_id: str,
    batch_job,
    batch_config: BatchGeminiConfig,
    db: Session,
    neo4j_session
):
    """
    Check the status of a batch job and proceed with the next steps when complete.
    
    Args:
        job_id: The unique identifier for this job
        batch_job: The batch job object
        batch_config: Configuration options for the pipeline steps
        db: Database session
        neo4j_session: Neo4j database session
    """
    try:
        gemini_client = BatchGeminiClient()
        
        # Special case: If the batch job name is "NO_PROCESSING_NEEDED", it means
        # all opinions already exist in Neo4j with AI summaries, so we can skip
        # the Gemini processing step
        if batch_job.name == "NO_PROCESSING_NEEDED":
            logger.info(f"[JOB:{job_id}] No processing needed - all opinions already exist in Neo4j")
            
            job_store[job_id].update({
                "status": JobStatus.COMPLETED,
                "updated_at": pd.Timestamp.now().isoformat(),
                "message": "No Gemini processing needed - all opinions already exist in Neo4j",
                "gemini_complete": True
            })
            
            # If cleaning is enabled, start the cleaning step
            if batch_config.run_cleaning:
                asyncio.create_task(
                    clean_pipeline_data(
                        job_id,
                        batch_config,
                        db,
                        neo4j_session
                    )
                )
            
            return
        
        while True:
            updated_job = gemini_client.check_batch_job_status(batch_job)
            
            # Map the Gemini job state to our job status
            gemini_state = updated_job.state
            mapped_status = JobStatus.PROCESSING
            if gemini_state == "JOB_STATE_SUCCEEDED":
                mapped_status = JobStatus.COMPLETED
            elif gemini_state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                mapped_status = JobStatus.ERROR
            
            job_store[job_id].update({
                "status": mapped_status,
                "updated_at": pd.Timestamp.now().isoformat(),
                "message": f"Job status: {gemini_state}"
            })
            
            # If the job is completed, proceed to the next step
            if gemini_state == "JOB_STATE_SUCCEEDED":
                # Get the results
                results_df = gemini_client.retrieve_batch_job_results(updated_job)
                results_file = save_to_tmp(results_df, f"results_{job_id}.csv", as_json=False)
                
                job_store[job_id].update({
                    "status": JobStatus.COMPLETED,
                    "results_file": results_file,
                    "message": "Gemini processing completed",
                    "gemini_complete": True
                })
                
                # If cleaning is enabled, start the cleaning step
                if batch_config.run_cleaning:
                    asyncio.create_task(
                        clean_pipeline_data(
                            job_id,
                            batch_config,
                            db,
                            neo4j_session
                        )
                    )
                
                break
            elif gemini_state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                job_store[job_id].update({
                    "status": JobStatus.ERROR,
                    "error": f"Gemini job failed with state: {gemini_state}"
                })
                return
            
            # Wait before checking again
            await asyncio.sleep(300)
            
    except Exception as e:
        error_message = f"Error checking pipeline job status: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })

async def clean_pipeline_data(
    job_id: str,
    batch_config: BatchGeminiConfig,
    db: Session,
    neo4j_session
):
    """
    Clean the data from the Gemini processing step.
    
    Args:
        job_id: The unique identifier for this job
        batch_config: Configuration options for the pipeline steps
        db: Database session
        neo4j_session: Neo4j database session
    """
    try:
        logger.info(f"[JOB:{job_id}] Starting data cleaning")
        job_store[job_id].update({
            "status": JobStatus.CLEANING,
            "message": "Cleaning data"
        })
        
        # Read the results file
        results_file = job_store[job_id]["results_file"]
        df = pd.read_csv(results_file)
        
        # Clean the data using the pipeline service function
        logger.info(f"[JOB:{job_id}] Cleaning data with {len(df)} rows")
        cleaned_df = pipeline_service.clean_extracted_opinions(df)
        logger.info(f"[JOB:{job_id}] Used clean_extracted_opinions")
        
        # Save the cleaned data
        cleaned_file = save_to_tmp(cleaned_df, f"cleaned_{job_id}.csv", as_json=False)
        
        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "cleaned_file": cleaned_file,
            "message": "Data cleaning completed",
            "cleaning_complete": True
        })
        
        # If resolution is enabled, start the resolution step
        if batch_config.run_resolution:
            asyncio.create_task(
                resolve_pipeline_citations(
                    job_id,
                    batch_config,
                    db,
                    neo4j_session
                )
            )
            
    except Exception as e:
        error_message = f"Error cleaning pipeline data: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })

async def resolve_pipeline_citations(
    job_id: str,
    batch_config: BatchGeminiConfig,
    db: Session,
    neo4j_session
):
    """
    Resolve citations from the cleaned data.
    
    Args:
        job_id: The unique identifier for this job
        batch_config: Configuration options for the pipeline steps
        db: Database session
        neo4j_session: Neo4j database session
    """
    try:
        logger.info(f"[JOB:{job_id}] Starting citation resolution")
        job_store[job_id].update({
            "status": JobStatus.RESOLVING,
            "message": "Resolving citations"
        })
        
        # Read the cleaned file
        cleaned_file = job_store[job_id]["cleaned_file"]
        df = pd.read_csv(cleaned_file)
        

        # Convert DataFrame to dictionary of CitationAnalysis objects
        llm_results = {}
        for _, row in df.iterrows():
            try:
                # Extract the cluster_id and create a CitationAnalysis object
                cluster_id = str(row.get('cluster_id', ''))
                if not cluster_id:
                    continue
                
                # Create a CitationAnalysis object from the row data
                analysis = CitationAnalysis(
                    date=row.get('date', '2023-01-01'),  # Default date if not available
                    brief_summary=row.get('ai_summary', ''),
                    majority_opinion_citations=[],  # These would need proper conversion from JSON
                    concurring_opinion_citations=[],  # These would need proper conversion from JSON
                    dissenting_citations=[]  # These would need proper conversion from JSON
                )
                llm_results[cluster_id] = [analysis]
            except Exception as e:
                logger.warning(f"[JOB:{job_id}] Error processing row for citation resolution: {str(e)}")
        
        # Validate and process the LLM results
        # Create combined analyses
        resolved_citations, _ = create_combined_analyses(llm_results)
        
        # Serialize and save the citations
        if resolved_citations:
            resolved_file = serialize_and_save_citations(resolved_citations)
        else:
            # Handle empty citations
            resolved_file = handle_empty_citations()
        
        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "resolved_file": resolved_file,
            "message": "Citation resolution completed",
            "resolution_complete": True
        })
        
        # If Neo4j loading is enabled, start the loading step
        if batch_config.run_neo4j_load:
            asyncio.create_task(
                load_pipeline_neo4j(
                    job_id,
                    batch_config,
                    db,
                    neo4j_session
                )
            )
            
    except Exception as e:
        error_message = f"Error resolving pipeline citations: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })

async def load_pipeline_neo4j(
    job_id: str,
    batch_config: BatchGeminiConfig,
    db: Session,
    neo4j_session
):
    """
    Load the resolved citations into Neo4j.
    
    Args:
        job_id: The unique identifier for this job
        batch_config: Configuration options for the pipeline steps
        db: Database session
        neo4j_session: Neo4j database session
    """
    try:
        logger.info(f"[JOB:{job_id}] Starting Neo4j loading")
        job_store[job_id].update({
            "status": JobStatus.LOADING_NEO4J,
            "message": "Loading to Neo4j"
        })
        
        # Read the resolved file
        resolved_file = job_store[job_id]["resolved_file"]
        
        # Create Neo4j loader
        loader = neomodel_loader
        
        # Load the resolved citations
        with open(resolved_file, 'r') as f:
            resolved_data = json.load(f)
        
        # Load each analysis into Neo4j
        for analysis_data in resolved_data:
            try:
                # Convert dict back to model
                analysis = CombinedResolvedCitationAnalysis(**analysis_data)
                
                # Load the analysis into Neo4j
                loader.load_enriched_citations([analysis], data_source="gemini_api")
            except Exception as e:
                logger.warning(f"[JOB:{job_id}] Error loading analysis into Neo4j: {str(e)}")
                # Continue with the next analysis
        
        # Final status update
        logger.info(f"[JOB:{job_id}] Full pipeline completed successfully")
        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "neo4j_loaded": True,
            "message": "Full pipeline completed successfully",
            "neo4j_complete": True
        })
        
    except Exception as e:
        error_message = f"Error loading pipeline data to Neo4j: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })

@router.post(
    "/submit-with-filters", 
    response_model=BatchGeminiJob, 
    dependencies=[Depends(verify_api_key)]
)
async def submit_batch_job_with_filters(
    background_tasks: BackgroundTasks,
    config: ExtractionConfig,
    db: Session = Depends(get_db)
):
    """
    Submit a batch job for Gemini processing using database filters.
    
    This endpoint aligns with the pipeline service by accepting the same
    filter parameters (ExtractionConfig) instead of requiring a CSV upload.
    
    Args:
        background_tasks: FastAPI background tasks
        config: Extraction configuration with filters
        db: Database session
        
    Returns:
        A JSON object with the assigned job ID and job status
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"[JOB:{job_id}] New job submission request received with filters: {config}")
    
    try:
        # Create a new job entry
        now = pd.Timestamp.now().isoformat()
        job_store[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "created_at": now,
            "config": config.model_dump(),
            "message": "Job queued for processing with database filters"
        }
        
        logger.info(f"[JOB:{job_id}] Created job entry with filters")
        
        # Start processing in the background
        background_tasks.add_task(
            process_batch_job_with_filters,
            job_id,
            config,
            db
        )
        
        return job_store[job_id]
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

def process_batch_job_with_filters(
    job_id: str,
    config: ExtractionConfig,
    db: Session
):
    """
    Process a batch job with Gemini using database filters.
    
    Args:
        job_id: The unique identifier for this job
        config: Extraction configuration with filters
        db: Database session
    """
    try:
        logger.info(f"[JOB:{job_id}] Starting batch job with filters")
        
        # Update job status
        job_store[job_id].update({
            "status": JobStatus.PROCESSING,
            "message": "Extracting opinions from database"
        })
        
        # Create the BatchGeminiClient
        gemini_client = BatchGeminiClient()
        
        # Generate a JSONL file name
        jsonl_file_name = f"batch_{job_id}.jsonl"
        
        # Extract opinions, clean them, and submit the batch job
        batch_job = gemini_client.submit_batch_job_from_config(
            config, 
            db.bind,  # Pass the database connection
            jsonl_file_name
        )
        
        # Update job status
        logger.info(f"[JOB:{job_id}] Submitted batch job to Gemini: {batch_job.name}")
        job_store[job_id].update({
            "status": JobStatus.SUBMITTED,
            "batch_job_name": batch_job.name,
            "message": f"Batch job submitted to Gemini: {batch_job.name}"
        })
        
    except Exception as e:
        error_message = f"Error processing batch job with filters: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })

def check_batch_job_exists(batch_job_name: str) -> bool:
    """
    Check if a Gemini batch job exists.
    
    Args:
        batch_job_name: The name of the batch job to check.
        
    Returns:
        True if the job exists, False otherwise.
    """
    try:
        gemini_client = BatchGeminiClient()
        batch_job = BatchJob(name=batch_job_name)
        gemini_client.check_batch_job_status(batch_job)
        return True
    except Exception as e:
        logger.error(f"Error checking if batch job exists: {str(e)}")
        return False

@router.post(
    "/continue-polling",
    response_model=BatchGeminiJob,
    dependencies=[Depends(verify_api_key)]
)
async def continue_polling_endpoint(
    background_tasks: BackgroundTasks,
    request: ContinuePollRequest
):
    """
    Continue polling for a Gemini batch job that may not be in our local job store.
    
    Args:
        background_tasks: FastAPI background tasks.
        request: The request containing the batch job name to poll.
        
    Returns:
        A JSON object containing the recreated job.
    """
    # Generate a new job ID if not provided
    job_id = request.original_job_id or str(uuid.uuid4())
    logger.info(f"[JOB:{job_id}] Continue polling request received for batch job: {request.batch_job_name}")
    
    # Check if the batch job exists
    if not check_batch_job_exists(request.batch_job_name):
        error_message = f"Batch job {request.batch_job_name} does not exist or cannot be accessed"
        logger.error(f"[JOB:{job_id}] {error_message}")
        raise HTTPException(status_code=404, detail=error_message)
    
    # Create a new job entry in the job store
    job_store[job_id] = {
        "job_id": job_id,
        "status": JobStatus.SUBMITTED,  # Start with SUBMITTED status
        "created_at": pd.Timestamp.now().isoformat(),
        "batch_job_name": request.batch_job_name,
        "message": f"Continuing polling for batch job: {request.batch_job_name}"
    }
    
    # Start polling for the job status
    gemini_client = BatchGeminiClient()
    background_tasks.add_task(check_job_status_background, gemini_client, job_id)
    
    # Return the job information
    return job_store[job_id]

@router.post(
    "/continue-pipeline",
    response_model=BatchGeminiJob,
    dependencies=[Depends(verify_api_key)]
)
async def continue_pipeline_endpoint(
    background_tasks: BackgroundTasks,
    request: ContinuePipelineRequest,
    db: Session = Depends(get_db),
    neo4j_session = Depends(get_neo4j)
):
    """
    Continue a pipeline that was interrupted, starting from the Gemini batch job.
    
    Args:
        background_tasks: FastAPI background tasks.
        request: The request containing the batch job name and pipeline configuration.
        db: Database session.
        neo4j_session: Neo4j database session.
        
    Returns:
        A JSON object containing the recreated job.
    """
    # Generate a new job ID if not provided
    job_id = request.original_job_id or str(uuid.uuid4())
    logger.info(f"[JOB:{job_id}] Continue pipeline request received for batch job: {request.batch_job_name}")
    
    # Check if the batch job exists
    if not check_batch_job_exists(request.batch_job_name):
        error_message = f"Batch job {request.batch_job_name} does not exist or cannot be accessed"
        logger.error(f"[JOB:{job_id}] {error_message}")
        raise HTTPException(status_code=404, detail=error_message)
    
    # Create a new job entry in the job store
    job_store[job_id] = {
        "job_id": job_id,
        "status": JobStatus.SUBMITTED,  # Start with SUBMITTED status
        "created_at": pd.Timestamp.now().isoformat(),
        "batch_job_name": request.batch_job_name,
        "message": f"Continuing pipeline for batch job: {request.batch_job_name}"
    }
    
    # Create a batch config from the request
    batch_config = BatchGeminiConfig(
        run_cleaning=request.run_cleaning,
        run_resolution=request.run_resolution,
        run_neo4j_load=request.run_neo4j_load
    )
    
    # Start the pipeline processing
    try:
        # Create a BatchJob object with the name
        batch_job = BatchJob(name=request.batch_job_name)
        
        # Start checking the job status
        background_tasks.add_task(
            check_pipeline_job_status,
            job_id,
            batch_job,
            batch_config,
            db,
            neo4j_session
        )
        
        return job_store[job_id]
    except Exception as e:
        error_message = f"Error starting pipeline continuation: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": JobStatus.ERROR,
            "error": error_message
        })
        raise HTTPException(status_code=500, detail=error_message) 