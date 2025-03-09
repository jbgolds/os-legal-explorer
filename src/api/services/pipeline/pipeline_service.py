import os
import json
import logging
from eyecite import clean_text
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union, Callable, TypeVar, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from src.llm_extraction.models import CitationAnalysis
from pydantic import TypeAdapter, BaseModel
from typing import List
from neomodel import db
from src.neo4j_db.models import Opinion

# Use the consolidated citation parser
from .pipeline_model import JobStatus, JobType, ExtractionConfig
from src.llm_extraction.rate_limited_gemini import GeminiClient
from src.llm_extraction.models import (
    CitationAnalysis,
    CombinedResolvedCitationAnalysis,
)
from src.neo4j_db.neomodel_loader import NeomodelLoader

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
        status: str = JobStatus.QUEUED,
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


# Helper functions for common operations
def save_to_tmp(
    data: Any, filename_prefix: str, as_json: bool = True, ensure_ascii: bool = False
) -> str:
    """
    Save data to a temporary file with timestamp.

    Args:
        data: Data to save
        filename_prefix: Prefix for the filename
        as_json: Whether to save as JSON (True) or CSV (False)
        ensure_ascii: Whether to escape non-ASCII characters in JSON

    Returns:
        Path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.{'json' if as_json else 'csv'}"
    output_path = os.path.join("/tmp", filename)

    if as_json:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)
    else:
        # Assume pandas DataFrame for non-JSON data
        data.to_csv(output_path, index=False)

    return output_path


T = TypeVar("T")


def job_step(
    db: Session, job_id: int, step_name: str, progress: float, fn: Callable[[], T]
) -> T:
    """
    Execute a job step with automatic status updates.

    Args:
        db: Database session
        job_id: Job ID
        step_name: Name of the step
        progress: Progress value (0-100)
        fn: Function to execute during the step

    Returns:
        Result of the function execution
    """
    # Update status before executing
    update_job_status(
        db,
        job_id,
        JobStatus.PROCESSING,
        progress=progress,
        message=f"Starting {step_name}",
    )

    # Execute the function
    result = fn()

    # Return the result
    return result


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
        "result_path": job.result_path,
    }


def get_jobs(
    db: Session,
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
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
    jobs = jobs[offset : offset + limit]

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
            "result_path": job.result_path,
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
    result_path: Optional[str] = None,
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


# New helper function to clean extracted opinions
def clean_extracted_opinions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean opinions extracted from our internal database.
    
    IMPORTANT: This function is specifically designed for data extracted from our own database
    and assumes certain column names and data structures. DO NOT use this for cleaning
    data from external sources or uploaded files.
    
    Args:
        df: DataFrame containing opinions extracted from our database
        
    Returns:
        Cleaned DataFrame with standardized text fields
    """

    logger.info(f"Cleaning extracted opinions, original length: {len(df)}")
    logger.info(f"Original DataFrame columns: {list(df.columns)}")
    # Make a copy and reset index
    new_df = df.copy().reset_index(drop=True)

    # Rename column 'id' to 'docket_database_id' if exists
    if "id" in new_df.columns:
        new_df = new_df.rename(columns={"id": "docket_database_id"})

    # Add new blank columns
    new_df["text"] = ""
    new_df["text_source"] = ""

    # Process each row; prioritize so_html_with_citations, then so_html, then so_plain_text
    for i, row in new_df.iterrows():
        if (
            pd.notna(row.get("so_html_with_citations"))
            and row.get("so_html_with_citations") != ""
        ):
            try:
                # Remove XML declaration before processing
                html_content = row["so_html_with_citations"].replace("<?xml version=\"1.0\" encoding=\"utf-8\"?>", "")
                new_df.at[i, "text"] = clean_text(html_content, ["html"])
                new_df.at[i, "text_source"] = "so_html_with_citations"
            except Exception as e:
                logger.error(f"Error cleaning so_html_with_citations: {e} for row {i} cluster_id {row['cluster_id']} with text {row['so_html_with_citations'][:100]}")
                raise e
            
        elif pd.notna(row.get("so_html")) and row.get("so_html") != "":
            try:
                # Remove XML declaration before processing
                html_content = row["so_html"].replace("<?xml version=\"1.0\" encoding=\"utf-8\"?>", "")
                new_df.at[i, "text"] = clean_text(html_content, ["html"])
                new_df.at[i, "text_source"] = "so_html"
            except Exception as e:
                logger.error(f"Error cleaning so_html: {e} for row {i} cluster_id {row['cluster_id']} with text {row['so_html'][:100]}")
                raise e

        elif pd.notna(row.get("so_plain_text")) and row.get("so_plain_text") != "":
            new_df.at[i, "text"] = row["so_plain_text"]
            new_df.at[i, "text_source"] = "so_plain_text"
        else:
            new_df.at[i, "text"] = ""
            new_df.at[i, "text_source"] = "no_text"

    # Filter out rows with no text
    new_df = new_df[new_df["text_source"] != "no_text"]

    # Filter out cases with fewer than 100 words
    new_df = new_df[new_df["text"].str.split().str.len() > 100]

    # Filter out rows containing specific petition phrases
    petition_phrases = [
        "Certiorari denied",
        "Petition for writ of mandamus denied",
        "Petitions for rehearing denied.",
        "Petition for writ of habeas corpus denied",
    ]
    for phrase in petition_phrases:
        new_df = new_df[~new_df["text"].fillna("").str.contains(phrase, na=False)]

    if "soc_date_filed" in new_df.columns:
        new_df["soc_date_filed"] = pd.to_datetime(
            new_df["soc_date_filed"], errors="coerce"
        )

        new_df = new_df.sort_values(by="soc_date_filed", ascending=False)

    # Remove duplicates based on case name, docket number, and date
    if all(
        col in new_df.columns
        for col in [
            "cluster_case_name",
            "sd_docket_number",
            "soc_date_filed",
            "soc_citation_count",
        ]
    ):
        # Group by identifying columns
        grouped = new_df.groupby(
            ["cluster_case_name", "sd_docket_number", "soc_date_filed"]
        )

        rows_to_keep = []
        for _, group in grouped:
            if len(group) > 1:  # If we have duplicates
                citations = group["soc_citation_count"].values
                if np.sum(citations == 0) == len(citations) - 1 and max(citations) > 0:
                    # Keep only the version with citations
                    rows_to_keep.append(group[group["soc_citation_count"] > 0].index[0])
                else:
                    # If pattern doesn't match, keep all versions
                    rows_to_keep.extend(group.index)
            else:
                # If no duplicates, keep the single version
                rows_to_keep.extend(group.index)

        new_df = new_df.loc[rows_to_keep].sort_values(
            ["cluster_case_name", "soc_date_filed"]
        )

    new_df = new_df.reset_index(drop=True)
    return new_df


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
            message="Starting opinion extraction",
        )

        # Build SQL query based on configuration
        filters = []
        params = {}

        # Check if we're processing a single cluster ID
        if config.single_cluster_id:
            filters.append("soc.id = %(single_cluster_id)s")
            params["single_cluster_id"] = config.single_cluster_id
            logger.info(f"Extracting single cluster ID: {config.single_cluster_id}")

        if config.court_id:
            filters.append("sd.court_id = %(court_id)s")
            params["court_id"] = config.court_id

        if config.start_date:
            filters.append("soc.date_filed >= %(start_date)s")
            params["start_date"] = config.start_date

        if config.end_date:
            filters.append("soc.date_filed <= %(end_date)s")
            params["end_date"] = config.end_date

        filter_clause = " AND ".join(filters)
        if filter_clause:
            filter_clause = (
                f"WHERE {filter_clause} AND soc.precedential_status = 'Published'"
            )
        else:
            filter_clause = "WHERE soc.precedential_status = 'Published'"

        # Build the query; adding html fields for cleaning
        query = f"""
        SELECT  
            so.cluster_id as cluster_id, 
            so.type as so_type, 
            so.id as so_id, 
            so.page_count as so_page_count, 
            so.html_with_citations as so_html_with_citations, 
            so.html as so_html, 
            so.plain_text as so_plain_text, 
            soc.case_name as cluster_case_name,
            soc.date_filed as soc_date_filed,
            soc.citation_count as soc_citation_count,
            sd.court_id as court_id,
            sd.docket_number as sd_docket_number,
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

        logger.info(f"Query: {query} with params: {params}")
        # Execute query
        update_job_status(
            db,
            job_id,
            JobStatus.PROCESSING,
            progress=10.0,
            message="Executing database query",
        )

        # Execute the actual database query
        if db.bind is None:
            raise ValueError("db.bind is None, cannot execute SQL query")
        df = pd.read_sql(query, db.bind, params=params)

        # Save raw CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"extracted_opinions_raw_{timestamp}.csv"
        output_path = os.path.join("/tmp", output_file)

        df.to_csv(output_path, index=False)

        df_len = len(df)

        # Update job status with raw extraction complete
        update_job_status(
            db,
            job_id,
            JobStatus.COMPLETED,
            progress=100.0,
            message=f"Extracted {len(df)} opinions (raw)",
            result_path=output_path,
        )

        logger.info(
            f"Completed extraction job {job_id}, saved raw data to {output_path} with length {df_len}"
        )

    except Exception as e:
        import traceback

        logger.error(
            f"Error in extraction job {job_id}: {str(e)}\n{traceback.format_exc()}"
        )
        update_job_status(db, job_id, JobStatus.FAILED, error=str(e))


class NodeStatus(BaseModel):
    """Status of a node in Neo4j."""

    exists: bool
    has_citations: bool
    citation_count: int
    has_ai_summary: bool = False


def check_node_status(cluster_id: str) -> NodeStatus:
    """
    Check if a node exists in Neo4j, has outgoing citations, and has ai_summary.

    Args:
        cluster_id: The ID of the cluster to check

    Returns:
        NodeStatus object with:
        - exists: Whether the node exists
        - has_citations: Whether the node has outgoing citations
        - citation_count: Number of outgoing citations
        - has_ai_summary: Whether the node has an ai_summary field filled out
    """
    try:
        # Try to find the opinion by primary_id (cluster_id)
        opinion = Opinion.nodes.first_or_none(primary_id=cluster_id)

        if not opinion:
            return NodeStatus(
                exists=False,
                has_citations=False,
                citation_count=0,
                has_ai_summary=False,
            )

        # Count outgoing citations using neomodel relationship
        # Get all outgoing CITES relationships
        citation_count = len(opinion.cites.all())

        # Check if ai_summary is filled out
        has_ai_summary = bool(opinion.ai_summary)

        return NodeStatus(
            exists=True,
            has_citations=citation_count > 0,
            citation_count=citation_count,
            has_ai_summary=has_ai_summary,
        )

    except Exception as e:
        logger.error(f"Error checking node status: {e}")
        # Return "not found" status on any error
        return NodeStatus(
            exists=False, has_citations=False, citation_count=0, has_ai_summary=False
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
            message="Starting LLM processing",
        )

        # Get extraction job
        extraction_job = get_job(db, extraction_job_id)
        if extraction_job is None:
            logger.error(
                f"Extraction job {extraction_job_id} not found, aborting pipeline"
            )
            return
        if extraction_job["status"] != JobStatus.COMPLETED:
            logger.error(
                f"Extraction job {extraction_job_id} failed, aborting pipeline"
            )
            return

        # Load and clean data
        df = pd.read_csv(extraction_job["result_path"])
        logger.info(f"Columns in raw DataFrame: {list(df.columns)}")
        cleaned_df = job_step(
            db, job_id, "data cleaning", 20.0, lambda: clean_extracted_opinions(df)
        )
        logger.info(f"Columns in cleaned DataFrame: {list(cleaned_df.columns)}")

        # Initialize already_processed_df with the same columns as cleaned_df
        already_processed_df = pd.DataFrame(columns=cleaned_df.columns)

        # now go through the cleaned df and check if the cluster_id is already in Neo4j with ai_summary
        for index, row in cleaned_df.iterrows():
            cluster_id = row["cluster_id"]
            node_status = check_node_status(str(cluster_id))
            if node_status.exists and node_status.has_ai_summary:
                logger.info(
                    f"Cluster {cluster_id} already exists in Neo4j with ai_summary. Skipping LLM processing to save API requests."
                )
                already_processed_df = pd.concat(
                    [already_processed_df, pd.DataFrame([row])], ignore_index=True
                )

        # drop the already processed rows from the cleaned df
        cleaned_df = cleaned_df[
            ~cleaned_df["cluster_id"].isin(already_processed_df["cluster_id"])
        ]
        logger.info(
            f"Remaining opinions to process: {len(cleaned_df)}. Dropped {len(already_processed_df)} opinions that already exist in Neo4j"
        )

        # Save intermediate CSV
        cleaned_output_path = save_to_tmp(
            cleaned_df, "extracted_opinions_cleaned", as_json=False
        )
        logger.info(f"Saved cleaned opinions to {cleaned_output_path}")

        # Check if the dataframe is empty
        if len(cleaned_df) == 0:
            # Save an empty JSON object instead of a CSV file
            empty_results = {}
            output_path = save_to_tmp(
                empty_results, "llm_results", as_json=True, ensure_ascii=False
            )

            update_job_status(
                db,
                job_id,
                JobStatus.COMPLETED,
                progress=100.0,
                message="No opinions to process after cleaning",
                result_path=output_path,
            )
            logger.warning(f"Completed LLM job {job_id} with empty dataframe")
            return

        # Process with LLM
        gemini_client = job_step(
            db,
            job_id,
            "LLM client initialization",
            30.0,
            lambda: GeminiClient(
                api_key=os.environ["GEMINI_API_KEY"], rpm_limit=50, max_concurrent=25
            ),
        )

        # Ensure batch_size is at least 1 to avoid division by zero
        batch_size = max(1, len(cleaned_df) // 10)  # Use 10 batches or at least 1
        results = job_step(
            db,
            job_id,
            f"LLM processing of {len(cleaned_df)} opinions",
            40.0,
            lambda: gemini_client.process_dataframe(
                cleaned_df, text_column="text", max_workers=25, batch_size=1
            ),
        )

        # Prepare and save results
        results_dump = {
            str(k): v.model_dump() if v is not None else None
            for k, v in results.items()
        }
        output_path = save_to_tmp(results_dump, "llm_results", ensure_ascii=False)

        # Update job status to completed
        update_job_status(
            db,
            job_id,
            JobStatus.COMPLETED,
            progress=100.0,
            message=f"Processed {len(results)} opinions with LLM",
            result_path=output_path,
        )

        logger.info(f"Completed LLM job {job_id}, saved to {output_path}")

    except Exception as e:
        import traceback

        logger.error(f"Error in LLM job {job_id}: {str(e)}\n{traceback.format_exc()}")
        update_job_status(db, job_id, JobStatus.FAILED, error=str(e))


def validate_llm_job(llm_job: Optional[Dict[str, Any]], llm_job_id: int) -> bool:
    """
    Validate that the LLM job exists and is completed.
    
    Args:
        llm_job: The LLM job dictionary
        llm_job_id: The LLM job ID
        
    Returns:
        bool: True if the job is valid, False otherwise
    """
    if llm_job is None:
        logger.error(f"LLM job {llm_job_id} not found, aborting pipeline")
        return False
    if llm_job["status"] != JobStatus.COMPLETED:
        logger.error(f"LLM job {llm_job_id} failed, aborting pipeline")
        return False
    return True


def load_and_validate_llm_data(llm_job: Dict[str, Any]) -> Dict[str, List[CitationAnalysis]]:
    """
    Load and validate LLM results from a job result file.
    
    Args:
        llm_job: The LLM job dictionary
        
    Returns:
        Dict[str, List[CitationAnalysis]]: Validated results by cluster ID
        
    Raises:
        ValueError: If the file format is invalid or cannot be parsed
    """
    # Check if the result file is a CSV (empty dataframe case) or JSON
    if llm_job["result_path"].endswith(".csv"):
        raise ValueError(
            f"Got a CSV file ({llm_job['result_path']}), expected a JSON file. This may happen if the LLM job produced an empty result set but didn't save it in the correct format."
        )
    with open(llm_job["result_path"], "r", encoding="utf-8") as f:
        try:
            llm_json = json.load(f)
        except json.decoder.JSONDecodeError as e:
            raise ValueError(
                f"Failed to decode JSON from file {f.name}: {str(e)}"
            )

    # If the JSON is empty, return an empty dictionary
    if not llm_json:
        logger.info("LLM job result is an empty JSON object")
        return {}

    # Create type adapters for validation
    list_adapter = TypeAdapter(List[CitationAnalysis])
    single_adapter = TypeAdapter(CitationAnalysis)

    validated_results = {}
    for cluster_id, dumped in llm_json.items():
        try:
            if dumped is None:
                continue

            # Check if dumped is a string (JSON) or already a dictionary
            if isinstance(dumped, str):
                # Try parsing as a list first, then as a single object
                try:
                    validated = list_adapter.validate_json(dumped)
                except Exception:
                    single_obj = single_adapter.validate_json(dumped)
                    validated = [single_obj]
            elif isinstance(dumped, dict):
                # Direct dictionary validation
                try:
                    validated = [single_adapter.validate_python(dumped)]
                except Exception:
                    # Try as a list if single validation fails
                    validated = list_adapter.validate_python([dumped])
            else:
                logger.warning(
                    f"Unexpected format for cluster {cluster_id}: {type(dumped)}"
                )
                continue

            # Store validated results
            validated_results[cluster_id] = validated
        except Exception as e:
            logger.warning(
                f"Validation failed for cluster {cluster_id}: {str(e)}"
            )

    logger.info(f"Loaded and validated {len(validated_results)} LLM results")
    return validated_results


def create_combined_analyses(validated_llm_results: Dict[str, List[CitationAnalysis]]) -> Tuple[List[CombinedResolvedCitationAnalysis], Dict[str, Any]]:
    """
    Process validated LLM results and create combined analyses.
    
    Args:
        validated_llm_results: Dictionary of validated LLM results by cluster ID
        
    Returns:
        Tuple containing:
        - List of resolved citation analyses
        - Dictionary of errors by cluster ID
    """
    resolved = []
    errors = {}

    # If there are no validated results, return empty lists
    if not validated_llm_results:
        logger.info("No validated LLM results to process")
        return resolved, errors

    for cluster_id, analyses in validated_llm_results.items():
        if not analyses:
            continue

        try:
            # Filter and validate analyses
            filtered_analyses = []
            for analysis in analyses:
                if isinstance(analysis, CitationAnalysis):
                    filtered_analyses.append(analysis)
                else:
                    logger.warning(
                        f"Unexpected analysis type for cluster {cluster_id}: {type(analysis)}"
                    )

            # Use the from_citations method to create a combined resolved analysis
            if filtered_analyses:
                # This method handles the resolution of citations internally
                combined = CombinedResolvedCitationAnalysis.from_citations(
                    filtered_analyses, int(cluster_id)
                )
                resolved.append(combined)
            else:
                logger.warning(
                    f"No valid analyses for cluster {cluster_id} after filtering"
                )

        except Exception as e:
            logger.warning(
                f"Error creating combined analysis for cluster {cluster_id}: {str(e)}"
            )
            errors[str(cluster_id)] = {
                "error": f"Error in combined analysis: {str(e)}",
                "analyses_count": len(analyses),
            }

    # Log summary information
    logger.info(
        f"Processed {len(validated_llm_results)} clusters, created {len(resolved)} valid citations"
    )
    if errors:
        logger.warning(
            f"Encountered {len(errors)} errors during citation resolution"
        )

    return resolved, errors


def serialize_and_save_citations(resolved_citations: List[CombinedResolvedCitationAnalysis]) -> str:
    """
    Serialize and save resolved citations to a temporary file.
    
    Args:
        resolved_citations: List of resolved citation analyses
        
    Returns:
        str: Path to the saved file
    """
    # Serialize citations
    serialized_citations = [
        citation.model_dump()  # Use model_dump() instead of model_dump_json() to get dictionaries
        for citation in resolved_citations
        if citation is not None
        and isinstance(citation, CombinedResolvedCitationAnalysis)
    ]

    if len(serialized_citations) < len(resolved_citations):
        logger.warning(
            f"Filtered out {len(resolved_citations) - len(serialized_citations)} invalid citation objects during serialization"
        )

    # Save to temporary file
    output_path = save_to_tmp(
        serialized_citations, "resolved_citations", ensure_ascii=False
    )
    
    return output_path


def handle_empty_citations() -> str:
    """
    Handle the case when there are no resolved citations.
    
    Returns:
        str: Path to the saved empty results file
    """
    logger.info("No citations to resolve, completing job with empty result")
    empty_results = {}
    output_path = save_to_tmp(
        empty_results, "resolved_citations", ensure_ascii=False
    )
    return output_path


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
            message="Starting citation resolution",
        )

        # Get and validate LLM job
        llm_job = get_job(db, llm_job_id)
        if not validate_llm_job(llm_job, llm_job_id):
            update_job_status(
                db,
                job_id,
                JobStatus.FAILED,
                error=f"LLM job {llm_job_id} is invalid or not completed"
            )
            return

        # Load and validate LLM results
        try:
            # At this point, llm_job is guaranteed to be not None because validate_llm_job would have returned False otherwise
            assert llm_job is not None, "LLM job should not be None at this point"
            
            validated_llm_results = job_step(
                db, job_id, "data validation", 20.0, 
                lambda: load_and_validate_llm_data(llm_job)
            )
        except Exception as e:
            logger.error(f"Failed to load and validate LLM data: {str(e)}")
            update_job_status(
                db, job_id, JobStatus.FAILED, 
                error=f"Failed to load and validate LLM data: {str(e)}"
            )
            return

        # Process all validated results and create combined analyses
        resolved_citations, errors = job_step(
            db, job_id, "combining analyses", 50.0,
            lambda: create_combined_analyses(validated_llm_results)
        )

        # Log any errors encountered during resolution
        if errors:
            logger.warning(
                f"Encountered {len(errors)} errors during citation resolution"
            )
            logger.warning(errors)

        # Handle empty citations case
        if not resolved_citations:
            output_path = handle_empty_citations()
            update_job_status(
                db,
                job_id,
                JobStatus.COMPLETED,
                progress=100.0,
                message="No citations to resolve after processing",
                result_path=output_path,
            )
            return

        # Serialize and save results
        output_path = serialize_and_save_citations(resolved_citations)

        # Update job status to complete
        update_job_status(
            db,
            job_id,
            JobStatus.COMPLETED,
            progress=100.0,
            message=f"Resolved citations for {len(resolved_citations)} opinions",
            result_path=output_path,
        )

        logger.info(f"Completed resolution job {job_id}, saved to {output_path}")

    except Exception as e:
        import traceback

        logger.error(
            f"Error in resolution job {job_id}: {str(e)}\n{traceback.format_exc()}"
        )
        update_job_status(db, job_id, JobStatus.FAILED, error=str(e))


def run_neo4j_job(
    db: Session, neo4j_session, job_id: int, resolution_job_id: int
) -> None:
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
            message="Starting Neo4j loading",
        )

        # Get resolution job
        resolution_job = get_job(db, resolution_job_id)
        if not resolution_job:
            raise ValueError(f"Resolution job {resolution_job_id} not found")

        # Check if the job has a result path
        if not resolution_job.get("result_path"):
            raise ValueError(f"Resolution job {resolution_job_id} has no result path")

        # Load resolution job result
        with open(resolution_job["result_path"], "r", encoding="utf-8") as f:
            resolved_citations = json.load(f)

        # Check if there are any resolved citations
        if not resolved_citations:
            logger.info(f"No citations to load from resolution job {resolution_job_id}")
            update_job_status(
                db,
                job_id,
                JobStatus.COMPLETED,
                progress=100.0,
                message="No citations to load into Neo4j",
                result_path=resolution_job["result_path"],
            )
            return

        # Convert to list of CombinedResolvedCitationAnalysis
        resolved_citations = [
            CombinedResolvedCitationAnalysis(**citation)
            for citation in resolved_citations
        ]

        logger.info(
            f"Loaded {len(resolved_citations)} citations from resolution job {resolution_job_id}"
        )

        # Update job status
        update_job_status(
            db,
            job_id,
            JobStatus.PROCESSING,
            progress=30.0,
            message="Initializing Neo4j loader",
        )

        loader = NeomodelLoader(
            url=os.environ["DB_NEO4J_URL"],
            username=os.environ["DB_NEO4J_USER"],
            password=os.environ["DB_NEO4J_PASSWORD"],
        )

        # Load citations into Neo4j
        update_job_status(
            db,
            job_id,
            JobStatus.PROCESSING,
            progress=50.0,
            message="Loading citations into Neo4j",
        )

        # DEBUG: Additional debugging for what we're passing to the loader
        for i, citation in enumerate(resolved_citations):
            if not hasattr(citation, "cluster_id"):
                logger.error(
                    f"Citation at index {i} has no cluster_id attribute: {type(citation)}"
                )
                if isinstance(citation, list):
                    logger.error(f"It's a list of length {len(citation)}")
                    if len(citation) > 0:
                        logger.error(f"First element type: {type(citation[0])}")

        loader.load_enriched_citations(resolved_citations, data_source="gemini_api")

        # Update job status
        update_job_status(
            db,
            job_id,
            JobStatus.COMPLETED,
            progress=100.0,
            message=f"Loaded {len(resolved_citations)} opinions into Neo4j",
        )

        logger.info(f"Completed Neo4j job {job_id}")

    except Exception as e:
        import traceback

        logger.error(f"Error in Neo4j job {job_id}: {str(e)}\n{traceback.format_exc()}")
        update_job_status(db, job_id, JobStatus.FAILED, error=str(e))


def run_full_pipeline(
    db: Session,
    neo4j_session,
    extraction_job_id: int,
    llm_job_id: int,
    resolution_job_id: int,
    neo4j_job_id: int,
    config: ExtractionConfig,
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
        if extraction_job is None:
            logger.error(
                f"Extraction job {extraction_job_id} not found, aborting pipeline"
            )
            return
        if extraction_job["status"] != JobStatus.COMPLETED:
            logger.error(
                f"Extraction job {extraction_job_id} failed, aborting pipeline"
            )
            return

        # Run LLM job
        run_llm_job(db, llm_job_id, extraction_job_id)

        # Check if LLM job succeeded
        llm_job = get_job(db, llm_job_id)
        if llm_job is None:
            logger.error(f"LLM job {llm_job_id} not found, aborting pipeline")
            return
        if llm_job["status"] != JobStatus.COMPLETED:
            logger.error(f"LLM job {llm_job_id} failed, aborting pipeline")
            return

        # Run resolution job
        run_resolution_job(db, resolution_job_id, llm_job_id)

        # Check if resolution job succeeded
        resolution_job = get_job(db, resolution_job_id)
        if resolution_job is None or resolution_job["status"] != JobStatus.COMPLETED:
            logger.error(
                f"Resolution job {resolution_job_id} failed, aborting pipeline"
            )
            return

        # Run Neo4j job
        run_neo4j_job(db, neo4j_session, neo4j_job_id, resolution_job_id)

        logger.info(f"Completed full pipeline")

    except Exception as e:
        logger.error(f"Error in full pipeline: {str(e)}")
