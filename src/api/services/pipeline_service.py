import os
import json
import logging
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
import time

from ..models.pipeline import JobStatus, JobType, ExtractionConfig
from src.llm_extraction.rate_limited_gemini import GeminiClient
from src.llm_extraction.models import CombinedResolvedCitationAnalysis
from src.neo4j.neomodel_loader import NeomodelLoader

logger = logging.getLogger(__name__)

# Define a simple job model for tracking pipeline jobs
# In a production environment, this would be a proper SQLAlchemy model
class PipelineJob:
    """Simple in-memory job model for tracking pipeline jobs."""
    
    def __init__(
        self, 
        job_id: int, 
        job_type: str, 
        config: Dict[str, Any],
        status: str = JobStatus.QUEUED
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.config = config
        self.status = status
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = 0.0
        self.message = None
        self.error = None
        self.result_path = None

# In-memory job storage (for demonstration)
# In a production environment, this would be stored in a database using SQLAlchemy models
_jobs = {}
_next_job_id = 1

def create_job(db: Session, job_type: str, config: Dict[str, Any]) -> int:
    """
    Create a new pipeline job.
    
    Args:
        db: Database session
        job_type: Type of job
        config: Job configuration
        
    Returns:
        Job ID
    """
    global _next_job_id
    
    # In a production environment, this would create a database record
    job_id = _next_job_id
    _next_job_id += 1
    
    job = PipelineJob(job_id, job_type, config)
    _jobs[job_id] = job
    
    logger.info(f"Created {job_type} job with ID {job_id}")
    return job_id

def get_job(db: Session, job_id: int) -> Optional[Dict[str, Any]]:
    """
    Get job status.
    
    Args:
        db: Database session
        job_id: Job ID
        
    Returns:
        Job status information
    """
    # In a production environment, this would query the database
    if job_id not in _jobs:
        return None
    
    job = _jobs[job_id]
    return {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "config": job.config,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "result_path": job.result_path
    }

def get_jobs(
    db: Session,
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get a list of jobs with optional filtering.
    
    Args:
        db: Database session
        job_type: Filter by job type
        status: Filter by job status
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        
    Returns:
        List of job status information
    """
    # In a production environment, this would query the database
    jobs = list(_jobs.values())
    
    # Apply filters
    if job_type:
        jobs = [job for job in jobs if job.job_type == job_type]
    
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda job: job.created_at, reverse=True)
    
    # Apply pagination
    jobs = jobs[offset:offset + limit]
    
    # Convert to dictionaries
    return [
        {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "config": job.config,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
            "result_path": job.result_path
        }
        for job in jobs
    ]

def update_job_status(
    db: Session,
    job_id: int,
    status: str,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    result_path: Optional[str] = None
) -> None:
    """
    Update job status.
    
    Args:
        db: Database session
        job_id: Job ID
        status: New job status
        progress: Job progress (0-100)
        message: Status message
        error: Error message
        result_path: Path to job result file
    """
    # In a production environment, this would update a database record
    if job_id not in _jobs:
        logger.warning(f"Attempted to update non-existent job {job_id}")
        return
    
    job = _jobs[job_id]
    job.status = status
    
    if progress is not None:
        job.progress = progress
    
    if message is not None:
        job.message = message
    
    if error is not None:
        job.error = error
    
    if result_path is not None:
        job.result_path = result_path
    
    # Update timestamps
    if status == JobStatus.STARTED and job.started_at is None:
        job.started_at = datetime.now()
    
    if status in [JobStatus.COMPLETED, JobStatus.FAILED] and job.completed_at is None:
        job.completed_at = datetime.now()
    
    logger.info(f"Updated job {job_id} status to {status}")

def run_extraction_job(db: Session, job_id: int, config: ExtractionConfig) -> None:
    """
    Run an opinion extraction job.
    
    Args:
        db: Database session
        job_id: Job ID
        config: Extraction configuration
    """
    try:
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.STARTED, 
            progress=0.0,
            message="Starting opinion extraction"
        )
        
        # Build SQL query based on configuration
        filters = []
        params = {}
        
        if config.court_id:
            filters.append("sd.court_id = :court_id")
            params["court_id"] = config.court_id
        
        if config.start_date:
            filters.append("soc.date_filed >= :start_date")
            params["start_date"] = config.start_date
        
        if config.end_date:
            filters.append("soc.date_filed <= :end_date")
            params["end_date"] = config.end_date
        
        filter_clause = " AND ".join(filters)
        if filter_clause:
            filter_clause = f"WHERE {filter_clause} AND soc.precedential_status = 'Published'"
        else:
            filter_clause = "WHERE soc.precedential_status = 'Published'"
        
        # Build the query
        query = f"""
        SELECT  
            so.cluster_id as cluster_id, 
            so.type as so_type, 
            so.id as so_id, 
            so.page_count as so_page_count, 
            so.plain_text as so_plain_text, 
            soc.case_name as cluster_case_name,
            soc.date_filed as soc_date_filed,
            sd.court_id as court_id,
            sc.full_name as court_name
        FROM search_opinion so 
        LEFT JOIN search_opinioncluster soc ON so.cluster_id = soc.id
        LEFT JOIN search_docket sd ON soc.docket_id = sd.id
        LEFT JOIN search_court sc ON sd.court_id = sc.id
        {filter_clause}
        ORDER BY soc.date_filed DESC
        """
        
        if config.limit:
            query += f" LIMIT {config.limit}"
        
        if config.offset:
            query += f" OFFSET {config.offset}"
        
        # Execute query
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=10.0,
            message="Executing database query"
        )
        
        # In a production environment, this would use the actual database connection
        # For demonstration, we'll simulate the query result
        # df = pd.read_sql(query, db.bind, params=params)
        
        # Simulate query result
        df = pd.DataFrame({
            "cluster_id": [1001, 1002, 1003],
            "so_type": ["010combined", "010combined", "010combined"],
            "so_id": [2001, 2002, 2003],
            "so_page_count": [10, 15, 20],
            "so_plain_text": [
                "This is the text of opinion 1001...",
                "This is the text of opinion 1002...",
                "This is the text of opinion 1003..."
            ],
            "cluster_case_name": [
                "Smith v. Jones",
                "Brown v. Board of Education",
                "Roe v. Wade"
            ],
            "soc_date_filed": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01"
            ],
            "court_id": ["scotus", "scotus", "scotus"],
            "court_name": [
                "Supreme Court of the United States",
                "Supreme Court of the United States",
                "Supreme Court of the United States"
            ]
        })
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"extracted_opinions_{timestamp}.csv"
        output_path = os.path.join("/tmp", output_file)
        
        df.to_csv(output_path, index=False)
        
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.COMPLETED, 
            progress=100.0,
            message=f"Extracted {len(df)} opinions",
            result_path=output_path
        )
        
        logger.info(f"Completed extraction job {job_id}, saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in extraction job {job_id}: {str(e)}")
        update_job_status(
            db, 
            job_id, 
            JobStatus.FAILED, 
            error=str(e)
        )

def run_llm_job(db: Session, job_id: int, extraction_job_id: int) -> None:
    """
    Run an LLM processing job.
    
    Args:
        db: Database session
        job_id: Job ID
        extraction_job_id: ID of the extraction job to process
    """
    try:
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.STARTED, 
            progress=0.0,
            message="Starting LLM processing"
        )
        
        # Get extraction job
        extraction_job = get_job(db, extraction_job_id)
        if not extraction_job:
            raise ValueError(f"Extraction job {extraction_job_id} not found")
        
        if extraction_job["status"] != JobStatus.COMPLETED:
            raise ValueError(f"Extraction job {extraction_job_id} is not completed")
        
        # Load extracted opinions
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=10.0,
            message="Loading extracted opinions"
        )
        
        # In a production environment, this would load the actual CSV file
        # For demonstration, we'll simulate the data
        # df = pd.read_csv(extraction_job["result_path"])
        
        # Simulate data
        df = pd.DataFrame({
            "cluster_id": [1001, 1002, 1003],
            "so_plain_text": [
                "This is the text of opinion 1001...",
                "This is the text of opinion 1002...",
                "This is the text of opinion 1003..."
            ]
        })
        
        # Initialize Gemini client
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=20.0,
            message="Initializing LLM client"
        )
        
        # In a production environment, this would use the actual Gemini client
        # For demonstration, we'll simulate the processing
        # gemini_client = GeminiClient(
        #     api_key=os.getenv("GEMINI_API_KEY"),
        #     rpm_limit=15,
        #     max_concurrent=10
        # )
        
        # Process opinions
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=30.0,
            message=f"Processing {len(df)} opinions with LLM"
        )
        
        # In a production environment, this would process the opinions with Gemini
        # For demonstration, we'll simulate the results
        # results = gemini_client.process_dataframe(
        #     df,
        #     text_column="so_plain_text",
        #     max_workers=10
        # )
        
        # Simulate results
        results = {
            1001: [{"parsed": {"date": "2020-01-01", "brief_summary": "Summary 1", "majority_opinion_citations": []}}],
            1002: [{"parsed": {"date": "2020-02-01", "brief_summary": "Summary 2", "majority_opinion_citations": []}}],
            1003: [{"parsed": {"date": "2020-03-01", "brief_summary": "Summary 3", "majority_opinion_citations": []}}]
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"llm_results_{timestamp}.json"
        output_path = os.path.join("/tmp", output_file)
        
        with open(output_path, "w") as f:
            json.dump(results, f)
        
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.COMPLETED, 
            progress=100.0,
            message=f"Processed {len(results)} opinions with LLM",
            result_path=output_path
        )
        
        logger.info(f"Completed LLM job {job_id}, saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in LLM job {job_id}: {str(e)}")
        update_job_status(
            db, 
            job_id, 
            JobStatus.FAILED, 
            error=str(e)
        )

def run_resolution_job(db: Session, job_id: int, llm_job_id: int) -> None:
    """
    Run a citation resolution job.
    
    Args:
        db: Database session
        job_id: Job ID
        llm_job_id: ID of the LLM job to process
    """
    try:
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.STARTED, 
            progress=0.0,
            message="Starting citation resolution"
        )
        
        # Get LLM job
        llm_job = get_job(db, llm_job_id)
        if not llm_job:
            raise ValueError(f"LLM job {llm_job_id} not found")
        
        if llm_job["status"] != JobStatus.COMPLETED:
            raise ValueError(f"LLM job {llm_job_id} is not completed")
        
        # Load LLM results
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=10.0,
            message="Loading LLM results"
        )
        
        # In a production environment, this would load the actual JSON file
        # For demonstration, we'll simulate the data
        # with open(llm_job["result_path"], "r") as f:
        #     llm_results = json.load(f)
        
        # Simulate results
        llm_results = {
            "1001": [{"parsed": {"date": "2020-01-01", "brief_summary": "Summary 1", "majority_opinion_citations": []}}],
            "1002": [{"parsed": {"date": "2020-02-01", "brief_summary": "Summary 2", "majority_opinion_citations": []}}],
            "1003": [{"parsed": {"date": "2020-03-01", "brief_summary": "Summary 3", "majority_opinion_citations": []}}]
        }
        
        # Resolve citations
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=30.0,
            message="Resolving citations"
        )
        
        # In a production environment, this would resolve the citations
        # For demonstration, we'll simulate the results
        resolved_citations = []
        for cluster_id, results in llm_results.items():
            try:
                # Create CombinedResolvedCitationAnalysis
                citation_analysis = CombinedResolvedCitationAnalysis.from_citations_json(
                    results, int(cluster_id)
                )
                resolved_citations.append(citation_analysis)
            except Exception as e:
                logger.warning(f"Error resolving citations for cluster {cluster_id}: {str(e)}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"resolved_citations_{timestamp}.json"
        output_path = os.path.join("/tmp", output_file)
        
        with open(output_path, "w") as f:
            json.dump(
                [citation.model_dump() for citation in resolved_citations],
                f
            )
        
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.COMPLETED, 
            progress=100.0,
            message=f"Resolved citations for {len(resolved_citations)} opinions",
            result_path=output_path
        )
        
        logger.info(f"Completed resolution job {job_id}, saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in resolution job {job_id}: {str(e)}")
        update_job_status(
            db, 
            job_id, 
            JobStatus.FAILED, 
            error=str(e)
        )

def run_neo4j_job(db: Session, neo4j_session, job_id: int, resolution_job_id: int) -> None:
    """
    Run a Neo4j loading job.
    
    Args:
        db: Database session
        neo4j_session: Neo4j session
        job_id: Job ID
        resolution_job_id: ID of the resolution job to process
    """
    try:
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.STARTED, 
            progress=0.0,
            message="Starting Neo4j loading"
        )
        
        # Get resolution job
        resolution_job = get_job(db, resolution_job_id)
        if not resolution_job:
            raise ValueError(f"Resolution job {resolution_job_id} not found")
        
        if resolution_job["status"] != JobStatus.COMPLETED:
            raise ValueError(f"Resolution job {resolution_job_id} is not completed")
        
        # Load resolved citations
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=10.0,
            message="Loading resolved citations"
        )
        
        # In a production environment, this would load the actual JSON file
        # For demonstration, we'll simulate the data
        # with open(resolution_job["result_path"], "r") as f:
        #     resolved_citations_data = json.load(f)
        #     resolved_citations = [
        #         CombinedResolvedCitationAnalysis.model_validate(data)
        #         for data in resolved_citations_data
        #     ]
        
        # Simulate data
        resolved_citations = [
            CombinedResolvedCitationAnalysis(
                date="2020-01-01",
                cluster_id=1001,
                brief_summary="Summary 1",
                majority_opinion_citations=[],
                concurrent_opinion_citations=[],
                dissenting_citations=[]
            ),
            CombinedResolvedCitationAnalysis(
                date="2020-02-01",
                cluster_id=1002,
                brief_summary="Summary 2",
                majority_opinion_citations=[],
                concurrent_opinion_citations=[],
                dissenting_citations=[]
            ),
            CombinedResolvedCitationAnalysis(
                date="2020-03-01",
                cluster_id=1003,
                brief_summary="Summary 3",
                majority_opinion_citations=[],
                concurrent_opinion_citations=[],
                dissenting_citations=[]
            )
        ]
        
        # Initialize Neo4j loader
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=30.0,
            message="Initializing Neo4j loader"
        )
        
        # In a production environment, this would use the actual Neo4j loader
        # For demonstration, we'll simulate the loading
        # loader = NeomodelLoader(
        #     uri=os.getenv("NEO4J_URI", "localhost:7687"),
        #     username=os.getenv("NEO4J_USER", "neo4j"),
        #     password=os.getenv("NEO4J_PASSWORD", "courtlistener"),
        #     database=os.getenv("NEO4J_DATABASE", "courtlistener")
        # )
        
        # Load citations
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=50.0,
            message="Loading citations into Neo4j"
        )
        
        # In a production environment, this would load the citations into Neo4j
        # For demonstration, we'll simulate the loading
        # loader.load_enriched_citations(resolved_citations, source="gemini_api")
        
        # Simulate loading
        time.sleep(1)
        
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.COMPLETED, 
            progress=100.0,
            message=f"Loaded {len(resolved_citations)} opinions into Neo4j"
        )
        
        logger.info(f"Completed Neo4j job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in Neo4j job {job_id}: {str(e)}")
        update_job_status(
            db, 
            job_id, 
            JobStatus.FAILED, 
            error=str(e)
        )

def process_uploaded_csv(db: Session, job_id: int, file_path: str) -> None:
    """
    Process an uploaded CSV file.
    
    Args:
        db: Database session
        job_id: Job ID
        file_path: Path to the uploaded CSV file
    """
    try:
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.STARTED, 
            progress=0.0,
            message="Starting CSV processing"
        )
        
        # Load CSV
        update_job_status(
            db, 
            job_id, 
            JobStatus.PROCESSING, 
            progress=10.0,
            message="Loading CSV file"
        )
        
        # In a production environment, this would load the actual CSV file
        # For demonstration, we'll simulate the data
        # df = pd.read_csv(file_path)
        
        # Simulate data
        df = pd.DataFrame({
            "cluster_id": [1001, 1002, 1003],
            "so_plain_text": [
                "This is the text of opinion 1001...",
                "This is the text of opinion 1002...",
                "This is the text of opinion 1003..."
            ]
        })
        
        # Create LLM job
        llm_job_id = create_job(
            db, 
            JobType.LLM_PROCESS, 
            {"file_path": file_path}
        )
        
        # Run LLM job
        run_llm_job(db, llm_job_id, job_id)
        
        # Update job status
        update_job_status(
            db, 
            job_id, 
            JobStatus.COMPLETED, 
            progress=100.0,
            message=f"Processed CSV file with {len(df)} opinions",
            result_path=file_path
        )
        
        logger.info(f"Completed CSV processing job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in CSV processing job {job_id}: {str(e)}")
        update_job_status(
            db, 
            job_id, 
            JobStatus.FAILED, 
            error=str(e)
        )

def run_full_pipeline(
    db: Session,
    neo4j_session,
    extraction_job_id: int,
    llm_job_id: int,
    resolution_job_id: int,
    neo4j_job_id: int,
    config: ExtractionConfig
) -> None:
    """
    Run the full pipeline from extraction to Neo4j loading.
    
    Args:
        db: Database session
        neo4j_session: Neo4j session
        extraction_job_id: ID of the extraction job
        llm_job_id: ID of the LLM job
        resolution_job_id: ID of the resolution job
        neo4j_job_id: ID of the Neo4j job
        config: Extraction configuration
    """
    try:
        # Run extraction job
        run_extraction_job(db, extraction_job_id, config)
        
        # Check if extraction job succeeded
        extraction_job = get_job(db, extraction_job_id)
        if extraction_job["status"] != JobStatus.COMPLETED:
            logger.error(f"Extraction job {extraction_job_id} failed, aborting pipeline")
            return
        
        # Run LLM job
        run_llm_job(db, llm_job_id, extraction_job_id)
        
        # Check if LLM job succeeded
        llm_job = get_job(db, llm_job_id)
        if llm_job["status"] != JobStatus.COMPLETED:
            logger.error(f"LLM job {llm_job_id} failed, aborting pipeline")
            return
        
        # Run resolution job
        run_resolution_job(db, resolution_job_id, llm_job_id)
        
        # Check if resolution job succeeded
        resolution_job = get_job(db, resolution_job_id)
        if resolution_job["status"] != JobStatus.COMPLETED:
            logger.error(f"Resolution job {resolution_job_id} failed, aborting pipeline")
            return
        
        # Run Neo4j job
        run_neo4j_job(db, neo4j_session, neo4j_job_id, resolution_job_id)
        
        logger.info(f"Completed full pipeline")
        
    except Exception as e:
        logger.error(f"Error in full pipeline: {str(e)}")
