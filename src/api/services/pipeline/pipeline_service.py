import os
import json
import logging
from eyecite import clean_text
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union, Callable, TypeVar, cast
from datetime import datetime
from sqlalchemy.orm import Session
from src.llm_extraction.models import CitationAnalysis, Citation, resolve_citation
from pydantic import TypeAdapter
from typing import List

# Use the consolidated citation parser
from .pipeline_model import JobStatus, JobType, ExtractionConfig
from src.llm_extraction.rate_limited_gemini import GeminiClient
from src.llm_extraction.models import (
    Citation,
    CitationAnalysis,
    CitationResolved,
    CombinedResolvedCitationAnalysis,
    resolve_citation,
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

    logger.info(f"Cleaning extracted opinions, original length: {len(df)}")
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
            new_df.at[i, "text"] = clean_text(row["so_html_with_citations"], ["html"])
            new_df.at[i, "text_source"] = "so_html_with_citations"
        elif pd.notna(row.get("so_html")) and row.get("so_html") != "":
            new_df.at[i, "text"] = clean_text(row["so_html"], ["html"])
            new_df.at[i, "text_source"] = "so_html"
        elif pd.notna(row.get("so_plain_text")) and row.get("so_plain_text") != "":
            new_df.at[i, "text"] = row["so_plain_text"]
            new_df.at[i, "text_source"] = "so_plain_text"
        else:
            new_df.at[i, "text"] = ""
            new_df.at[i, "text_source"] = "no_text"

    # Filter out rows with no text
    new_df = new_df[new_df["text_source"] != "no_text"]

    # Filter out cases with fewer than 250 words
    # new_df = new_df[new_df["text"].str.split().str.len() > 100]

    # # Filter out rows containing specific petition phrases
    # petition_phrases = [
    #     "Certiorari denied",
    #     "Petition for writ of mandamus denied",
    #     "Petitions for rehearing denied.",
    #     "Petition for writ of habeas corpus denied",
    # ]
    # for phrase in petition_phrases:
    #     new_df = new_df[~new_df["text"].fillna("").str.contains(phrase, na=False)]

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
            f"Completed extraction job {job_id}, saved raw data to {output_path}"
        )

    except Exception as e:
        import traceback

        logger.error(
            f"Error in extraction job {job_id}: {str(e)}\n{traceback.format_exc()}"
        )
        update_job_status(db, job_id, JobStatus.FAILED, error=str(e))


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
        cleaned_df = job_step(
            db, job_id, "data cleaning", 20.0, lambda: clean_extracted_opinions(df)
        )

        # Save intermediate CSV
        cleaned_output_path = save_to_tmp(
            cleaned_df, "extracted_opinions_cleaned", as_json=False
        )
        logger.info(f"Saved cleaned opinions to {cleaned_output_path}")

        # Check if the dataframe is empty
        if len(cleaned_df) == 0:
            update_job_status(
                db,
                job_id,
                JobStatus.COMPLETED,
                progress=100.0,
                message="No opinions to process after cleaning",
                result_path=cleaned_output_path,
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
                api_key=os.environ["GEMINI_API_KEY"], rpm_limit=15, max_concurrent=10
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
                cleaned_df, text_column="text", max_workers=10, batch_size=batch_size
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

        # Get LLM job
        llm_job = get_job(db, llm_job_id)
        if llm_job is None:
            logger.error(f"LLM job {llm_job_id} not found, aborting pipeline")
            return
        if llm_job["status"] != JobStatus.COMPLETED:
            logger.error(f"LLM job {llm_job_id} failed, aborting pipeline")
            return

        # Load and validate LLM results
        def load_and_validate_llm_data():
            with open(llm_job["result_path"], "r", encoding="utf-8") as f:
                llm_json = json.load(f)

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

        validated_llm_results = job_step(
            db, job_id, "data validation", 20.0, load_and_validate_llm_data
        )

        # Process all validated results and create combined analyses
        def create_combined_analyses():
            resolved = []
            errors = {}

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

        resolved_citations, _ = job_step(
            db, job_id, "combining analyses", 50.0, create_combined_analyses
        )

        # Serialize and save results
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

        output_path = save_to_tmp(
            serialized_citations, "resolved_citations", ensure_ascii=False
        )

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

        if resolution_job["status"] != JobStatus.COMPLETED:
            raise ValueError(f"Resolution job {resolution_job_id} is not completed")

        # Load resolved citations
        with open(resolution_job["result_path"], "r") as f:
            resolved_citations_data = json.load(f)
            logger.info(
                f"Loaded {len(resolved_citations_data)} citation entries from JSON"
            )
            logger.debug(
                f"Type of resolved_citations_data: {type(resolved_citations_data)}"
            )
            if isinstance(resolved_citations_data, list):
                preview = resolved_citations_data[:2]  # preview first two entries
            else:
                preview = resolved_citations_data
            logger.debug(f"Preview of resolved citations data: {preview}")

            # Safer validation with debugging
            resolved_citations = []
            skipped_items = 0
            for i, data in enumerate(resolved_citations_data):
                try:
                    # Parse the item if it's a string (from model_dump_json)
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            logger.error(f"Item {i} is not valid JSON: {data[:100]}")
                            skipped_items += 1
                            continue

                    # DEBUG: Print information about each item
                    if isinstance(data, list):
                        logger.error(
                            f"Item {i} is a list, not an object: {str(data)[:100]}"
                        )
                        skipped_items += 1
                        continue

                    # Check if required fields exist
                    if not isinstance(data, dict):
                        logger.error(f"Item {i} is not a dictionary: {type(data)}")
                        skipped_items += 1
                        continue

                    if "cluster_id" not in data:
                        logger.error(
                            f"Item {i} missing cluster_id: {list(data.keys())}"
                        )
                        skipped_items += 1
                        continue

                    # Check required fields for citations
                    required_fields = ["date", "brief_summary", "cluster_id"]
                    missing_fields = [
                        field for field in required_fields if field not in data
                    ]
                    if missing_fields:
                        logger.error(
                            f"Item {i} missing required fields: {missing_fields}"
                        )
                        skipped_items += 1
                        continue

                    # Try validation with traceback details
                    try:
                        citation = CombinedResolvedCitationAnalysis.model_validate(data)
                        resolved_citations.append(citation)
                    except Exception as e:
                        import traceback

                        logger.error(
                            f"Error validating item {i}: {str(e)}\n{traceback.format_exc()}"
                        )
                        try:
                            data_str = json.dumps(data, indent=2)
                        except Exception:
                            data_str = str(data)
                        logger.error(f"Item {i} data (truncated): {data_str[:200]}")
                        skipped_items += 1
                except Exception as exc:
                    import traceback

                    logger.error(
                        f"Unexpected error processing item {i}: {str(exc)}\n{traceback.format_exc()}"
                    )
                    skipped_items += 1

        # Log how many valid citations we found
        logger.info(
            f"Successfully validated {len(resolved_citations)} out of {len(resolved_citations_data)} citations (skipped {skipped_items})"
        )

        # Initialize Neo4j loader
        update_job_status(
            db,
            job_id,
            JobStatus.PROCESSING,
            progress=30.0,
            message="Initializing Neo4j loader",
        )

        loader = NeomodelLoader(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
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
            message="Starting CSV processing",
        )

        # Load CSV
        update_job_status(
            db, job_id, JobStatus.PROCESSING, progress=10.0, message="Loading CSV file"
        )

        # Load the actual CSV file
        df = pd.read_csv(file_path)

        # Create LLM job
        llm_job_id = create_job(db, JobType.LLM_PROCESS, {"file_path": file_path})

        # Run LLM job
        run_llm_job(db, llm_job_id, job_id)

        # Update job status
        update_job_status(
            db,
            job_id,
            JobStatus.COMPLETED,
            progress=100.0,
            message=f"Processed CSV file with {len(df)} opinions",
            result_path=file_path,
        )

        logger.info(f"Completed CSV processing job {job_id}")

    except Exception as e:
        logger.error(f"Error in CSV processing job {job_id}: {str(e)}")
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
