import os
import uuid
import logging
import asyncio
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, File, UploadFile, Query
from fastapi.security import APIKeyHeader
import pandas as pd
from pydantic import BaseModel

# Import the BatchGeminiClient
from src.llm_extraction.batch_gemini import BatchGeminiClient
from google.genai.types import BatchJob

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

# Define Pydantic models for request/response
class BatchGeminiJob(BaseModel):
    """Model representing a batch Gemini job."""
    job_id: str
    status: str
    created_at: str
    file_path: Optional[str] = None
    row_count: Optional[int] = None
    text_column: Optional[str] = None
    batch_job_name: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    
class BatchGeminiStatus(BaseModel):
    """Model representing the status of a batch Gemini job."""
    job_id: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    
class BatchGeminiConfig(BaseModel):
    """Configuration for batch Gemini processing."""
    text_column: str = "text"

# In-memory job storage (replace with a database in production)
job_store: Dict[str, Dict[str, Any]] = {}

def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify that the API key is valid."""
    logger.debug(f"Verifying API key: {'valid' if api_key == API_KEY else 'invalid'}")
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempted: {api_key[:5]}...")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

def save_to_tmp(data: pd.DataFrame, filename: str) -> str:
    """Save DataFrame to a temporary CSV file."""
    logger.debug(f"Saving DataFrame with {len(data)} rows to temporary file: {filename}")
    tmp_path = os.path.join("/tmp", filename)
    try:
        data.to_csv(tmp_path, index=False)
        logger.debug(f"Successfully saved data to {tmp_path}")
        return tmp_path
    except Exception as e:
        logger.exception(f"Error saving DataFrame to {tmp_path}: {str(e)}")
        raise

def process_batch_job(
    client: BatchGeminiClient, 
    df: pd.DataFrame, 
    job_id: str, 
    text_column: str
):
    """
    Background task that processes a batch job using BatchGeminiClient.
    
    This function runs outside the main request thread to avoid blocking
    the event loop.
    
    Args:
        client: An instance of BatchGeminiClient.
        df: DataFrame containing the input data.
        job_id: The unique identifier of the job.
        text_column: The column containing the text to analyze.
    """
    logger.info(f"[JOB:{job_id}] Starting batch job processing with {len(df)} rows")
    
    # Update job status to processing
    job_store[job_id]["status"] = "PROCESSING"
    logger.debug(f"[JOB:{job_id}] Status updated to PROCESSING")
    
    try:
        # Create a JSONL file name using the job ID
        jsonl_file_name = f"batch_job_{job_id}.jsonl"
        logger.debug(f"[JOB:{job_id}] Will use JSONL file: {jsonl_file_name}")
        
        # Submit the batch job - this is a blocking operation
        logger.info(f"[JOB:{job_id}] Submitting batch job to Gemini API")
        batch_job = client.submit_batch_job(df, jsonl_file_name, text_column=text_column)
        

        # Store batch job information
        logger.info(f"[JOB:{job_id}] Batch job submitted successfully with name: {batch_job.name}")
        job_store[job_id].update({
            "status": "SUBMITTED",
            "batch_job_name": batch_job.name,
            "message": f"Batch job submitted successfully: {batch_job.name}"
        })
        
        logger.debug(f"[JOB:{job_id}] Updated job store with batch job name")
    except Exception as e:
        # Update job status to error
        error_message = f"Error submitting batch job: {str(e)}"
        logger.exception(f"[JOB:{job_id}] {error_message}")
        job_store[job_id].update({
            "status": "ERROR",
            "error": error_message
        })
        logger.error(error_message)

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
                    "message": "Results retrieved successfully"
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

@router.post(
    "/submit", 
    response_model=BatchGeminiJob, 
    dependencies=[Depends(verify_api_key)]
)
async def submit_batch_job_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text_column: str = "text"
):
    """
    Submit a batch job for Gemini processing.
    
    Args:
        background_tasks: FastAPI background tasks.
        file: CSV file with input data.
        text_column: Column name containing the text to analyze.
        
    Returns:
        A JSON object with the assigned job ID and job status.
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"[JOB:{job_id}] New job submission request received for file: {file.filename}")
    
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("/tmp", f"upload_{job_id}.csv")
        logger.debug(f"[JOB:{job_id}] Saving uploaded file to {temp_file_path}")
        
        with open(temp_file_path, "wb") as buffer:
            file_content = await file.read()
            buffer.write(file_content)
            logger.debug(f"[JOB:{job_id}] File saved, size: {len(file_content)} bytes")
        
        # Read the CSV into a DataFrame
        logger.debug(f"[JOB:{job_id}] Reading CSV file into DataFrame")
        df = pd.read_csv(temp_file_path)
        logger.info(f"[JOB:{job_id}] DataFrame loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Log column names for debugging
        logger.debug(f"[JOB:{job_id}] Available columns: {', '.join(df.columns)}")
        
        # Validate that the text column exists in the DataFrame
        if text_column not in df.columns:
            error_msg = f"Text column '{text_column}' not found. Available columns: {', '.join(df.columns)}"
            logger.error(f"[JOB:{job_id}] {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Instantiate the BatchGeminiClient
        logger.debug(f"[JOB:{job_id}] Initializing BatchGeminiClient")
        gemini_client = BatchGeminiClient()
        
        # Initialize the job in the store
        logger.debug(f"[JOB:{job_id}] Creating job entry in store")
        job_store[job_id] = {
            "job_id": job_id,
            "status": "QUEUED",
            "created_at": pd.Timestamp.now().isoformat(),
            "file_path": temp_file_path,
            "row_count": len(df),
            "text_column": text_column
        }
        
        # Add the batch job processing task to background tasks
        logger.info(f"[JOB:{job_id}] Adding job to background tasks")
        background_tasks.add_task(
            process_batch_job, 
            gemini_client, 
            df, 
            job_id, 
            text_column
        )
        
        return job_store[job_id]
    except Exception as e:
        # If we've already created a job entry, update its status
        logger.exception(f"[JOB:{job_id}] Error during job submission: {str(e)}")
        if job_id in job_store:
            logger.debug(f"[JOB:{job_id}] Updating job store with error status")
            job_store[job_id].update({
                "status": "ERROR",
                "error": str(e)
            })
            return job_store[job_id]
        
        # Re-raise with HTTP exception
        logger.error(f"[JOB:{job_id}] Returning HTTP exception")
        raise HTTPException(status_code=500, detail=f"Error processing submission: {str(e)}")

@router.get(
    "/status/{job_id}", 
    response_model=BatchGeminiStatus, 
    dependencies=[Depends(verify_api_key)]
)
async def check_batch_job_status_endpoint(
    background_tasks: BackgroundTasks,
    job_id: str
):
    """
    Check the status of a submitted batch job.
    
    Args:
        background_tasks: FastAPI background tasks.
        job_id: The unique identifier of the batch job.
        
    Returns:
        A JSON object containing the job status.
    """
    logger.info(f"[JOB:{job_id}] Status check request received")
    
    if job_id not in job_store:
        logger.warning(f"[JOB:{job_id}] Job not found in job store")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Always check the latest status if we have a batch job name
    if "batch_job_name" in job_store[job_id]:
        logger.info(f"[JOB:{job_id}] Checking latest status")
        gemini_client = BatchGeminiClient()
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
        job_data.get("results_file")
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
    "/run-full-pipeline",
    response_model=BatchGeminiJob,
    dependencies=[Depends(verify_api_key)],
)
async def run_full_pipeline(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: BatchGeminiConfig = Depends()
):
    """
    Run the full batch Gemini pipeline in one operation:
    1. Upload and process the input file
    2. Submit the batch job to Gemini
    3. Monitor job status automatically
    4. Retrieve results when ready
    
    Args:
        background_tasks: FastAPI background tasks.
        file: CSV file with input data
        config: Configuration for the batch processing
        
    Returns:
        A JSON object with the job ID and initial status
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"[JOB:{job_id}] Full pipeline request received for file: {file.filename}")
    
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("/tmp", f"upload_{job_id}.csv")
        logger.debug(f"[JOB:{job_id}] Saving uploaded file to {temp_file_path}")
        
        with open(temp_file_path, "wb") as buffer:
            file_content = await file.read()
            buffer.write(file_content)
            logger.debug(f"[JOB:{job_id}] File saved, size: {len(file_content)} bytes")
        
        # Read the CSV into a DataFrame
        logger.debug(f"[JOB:{job_id}] Reading CSV file into DataFrame")
        df = pd.read_csv(temp_file_path)
        logger.info(f"[JOB:{job_id}] DataFrame loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Validate that the text column exists in the DataFrame
        text_column = config.text_column
        logger.debug(f"[JOB:{job_id}] Validating text column: {text_column}")
        if text_column not in df.columns:
            error_msg = f"Text column '{text_column}' not found. Available columns: {', '.join(df.columns)}"
            logger.error(f"[JOB:{job_id}] {error_msg}")
            raise HTTPException(
                status_code=400, 
                detail=error_msg
            )
        
        # Instantiate the BatchGeminiClient
        logger.debug(f"[JOB:{job_id}] Initializing BatchGeminiClient")
        gemini_client = BatchGeminiClient()
        
        # Initialize the job in the store
        logger.debug(f"[JOB:{job_id}] Creating job entry in store for full pipeline")
        job_store[job_id] = {
            "job_id": job_id,
            "status": "QUEUED",
            "created_at": pd.Timestamp.now().isoformat(),
            "file_path": temp_file_path,
            "row_count": len(df),
            "text_column": text_column,
            "message": "Full pipeline job queued"
        }
        
        # Create a function to execute the full pipeline
        async def execute_full_pipeline():
            logger.info(f"[JOB:{job_id}] Starting full pipeline execution")
            # Step 1: Submit the batch job
            try:
                # Create a JSONL file name using the job ID
                jsonl_file_name = f"batch_job_{job_id}.jsonl"
                
                # Update status
                logger.debug(f"[JOB:{job_id}] Updating status to PROCESSING")
                job_store[job_id]["status"] = "PROCESSING"
                job_store[job_id]["message"] = "Submitting batch job"
                
                # Submit the batch job - this is a blocking operation
                logger.info(f"[JOB:{job_id}] Submitting batch job to Gemini API")
                batch_job = gemini_client.submit_batch_job(df, jsonl_file_name, text_column=text_column)
                
                # Store batch job information
                logger.info(f"[JOB:{job_id}] Batch job submitted with name: {batch_job.name}")
                job_store[job_id].update({
                    "status": "SUBMITTED",
                    "batch_job_name": batch_job.name,
                    "message": f"Batch job submitted: {batch_job.name}"
                })
                
                # Step 2: Monitor the job until it's completed
                monitoring_attempts = 0
                max_monitoring_attempts = 60  # Limit the number of monitoring attempts
                
                logger.info(f"[JOB:{job_id}] Starting job monitoring loop (max {max_monitoring_attempts} attempts)")
                while monitoring_attempts < max_monitoring_attempts:
                    # Wait between status checks
                    logger.debug(f"[JOB:{job_id}] Waiting 30 seconds before next status check")
                    await asyncio.sleep(30)  # 30 seconds between checks
                    
                    # Check job status
                    try:
                        logger.info(f"[JOB:{job_id}] Checking job status (attempt {monitoring_attempts+1}/{max_monitoring_attempts})")
                        updated_job = gemini_client.check_batch_job_status(batch_job)
                        
                        logger.info(f"[JOB:{job_id}] Current job state: {updated_job.state}")
                        job_store[job_id].update({
                            "status": updated_job.state,
                            "updated_at": pd.Timestamp.now().isoformat(),
                            "message": f"Job status: {updated_job.state}"
                        })
                        
                        # If the job is completed, proceed to the next step
                        if updated_job.state == "JOB_STATE_SUCCEEDED":
                            # Step 3: Retrieve the results
                            logger.info(f"[JOB:{job_id}] Job succeeded, retrieving results")
                            job_store[job_id]["message"] = "Retrieving results"
                            results_df = gemini_client.retrieve_batch_job_results(updated_job)
                            
                            if results_df is not None:
                                logger.debug(f"[JOB:{job_id}] Results retrieved with {len(results_df)} rows")
                                # Save results to a temporary CSV file
                                results_file = save_to_tmp(results_df, f"results_{job_id}.csv")
                                
                                # Update job status with the results file path
                                logger.info(f"[JOB:{job_id}] Results saved to {results_file}")
                                job_store[job_id].update({
                                    "status": "COMPLETED",
                                    "results_file": results_file,
                                    "message": "Full pipeline completed successfully"
                                })
                                break  # Exit the monitoring loop
                            else:
                                logger.error(f"[JOB:{job_id}] Results retrieval returned None")
                                job_store[job_id].update({
                                    "status": "ERROR",
                                    "error": "Results retrieval returned None"
                                })
                                break  # Exit the monitoring loop
                        
                        # If the job failed, update the status and exit
                        if updated_job.state == "JOB_STATE_FAILED":
                            logger.error(f"[JOB:{job_id}] Batch job failed")
                            job_store[job_id].update({
                                "status": "ERROR",
                                "error": f"Batch job failed: {updated_job.state}"
                            })
                            break  # Exit the monitoring loop
                            
                        # Increment the monitoring attempt counter
                        monitoring_attempts += 1
                    except Exception as e:
                        logger.exception(f"[JOB:{job_id}] Error monitoring job: {str(e)}")
                        job_store[job_id].update({
                            "status": "ERROR",
                            "error": f"Error monitoring job: {str(e)}"
                        })
                        break  # Exit the monitoring loop
                
                # If we've reached the maximum number of monitoring attempts
                if monitoring_attempts >= max_monitoring_attempts:
                    logger.warning(f"[JOB:{job_id}] Job monitoring timed out after {max_monitoring_attempts} attempts")
                    job_store[job_id].update({
                        "status": "ERROR",
                        "error": "Job monitoring timed out"
                    })
            except Exception as e:
                # Update job status to error
                error_message = f"Error in full pipeline: {str(e)}"
                logger.exception(f"[JOB:{job_id}] {error_message}")
                job_store[job_id].update({
                    "status": "ERROR",
                    "error": error_message
                })
        
        # Add the full pipeline execution task to background tasks
        logger.info(f"[JOB:{job_id}] Adding full pipeline execution to background tasks")
        background_tasks.add_task(execute_full_pipeline)
        
        return job_store[job_id]
    except Exception as e:
        # If we've already created a job entry, update its status
        logger.exception(f"[JOB:{job_id}] Error during full pipeline setup: {str(e)}")
        if job_id in job_store:
            job_store[job_id].update({
                "status": "ERROR",
                "error": str(e)
            })
            return job_store[job_id]
        
        # Re-raise with HTTP exception
        raise HTTPException(status_code=500, detail=f"Error starting pipeline: {str(e)}") 