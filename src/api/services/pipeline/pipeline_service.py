import os
import json
import logging
from eyecite import clean_text
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session


# Use the consolidated citation parser
from .pipeline_model import JobStatus, JobType, ExtractionConfig
from src.llm_extraction.rate_limited_gemini import GeminiClient
from src.llm_extraction.models import (
    CombinedResolvedCitationAnalysis,
    CitationAnalysis,
    Citation,
)
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
                if (citations == 0).sum() == len(citations) - 1 and max(citations) > 0:
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
        logger.error(f"Error in extraction job {job_id}: {str(e)}")
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
        if not extraction_job:
            raise ValueError(f"Extraction job {extraction_job_id} not found")

        if extraction_job["status"] != JobStatus.COMPLETED:
            raise ValueError(f"Extraction job {extraction_job_id} is not completed")

        # Load extracted opinions CSV
        raw_csv_path = extraction_job["result_path"]
        df = pd.read_csv(raw_csv_path)

        # Clean and transform the dataframe using the helper function
        cleaned_df = clean_extracted_opinions(df)

        # Save cleaned CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_output_file = f"extracted_opinions_cleaned_{timestamp}.csv"
        cleaned_output_path = os.path.join("/tmp", cleaned_output_file)
        cleaned_df.to_csv(cleaned_output_path, index=False)

        # Update job status
        update_job_status(
            db,
            job_id,
            JobStatus.PROCESSING,
            progress=20.0,
            message=f"Cleaned extracted opinions: {len(cleaned_df)} records ready for LLM processing",
        )

        # Initialize Gemini client
        update_job_status(
            db,
            job_id,
            JobStatus.PROCESSING,
            progress=30.0,
            message="Initializing LLM client",
        )

        gemini_client = GeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"), rpm_limit=15, max_concurrent=10
        )

        # Process opinions using the cleaned text column
        update_job_status(
            db,
            job_id,
            JobStatus.PROCESSING,
            progress=40.0,
            message=f"Processing {len(cleaned_df)} opinions with LLM",
        )

        # Here we use the 'text' column from cleaned_df
        results = gemini_client.process_dataframe(
            cleaned_df, text_column="text", max_workers=10
        )

        # Note: The gemini_client already saves a comprehensive debug file with raw responses
        # and validation errors to /tmp/gemini_debug_TIMESTAMP.json

        # Save LLM results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"llm_results_{timestamp}.json"
        output_path = os.path.join("/tmp", output_file)

        # Convert CitationAnalysis objects to dictionaries before serialization
        serializable_results = {}
        for cluster_id, result_list in results.items():
            if result_list is not None:
                # Handle both single objects and lists
                if isinstance(result_list, list):
                    serializable_results[cluster_id] = [
                        item.model_dump() if hasattr(item, "model_dump") else item
                        for item in result_list
                    ]
                else:
                    serializable_results[cluster_id] = (
                        result_list.model_dump()
                        if hasattr(result_list, "model_dump")
                        else result_list
                    )

        # Preprocess LLM results to ensure required fields are present and normalized
        for cluster_id, results in serializable_results.items():
            if results:
                # If results is a list, process each citation dict
                if isinstance(results, list):
                    for item in results:
                        if isinstance(item, dict):
                            if "relevance" in item and item["relevance"] == "":
                                item["relevance"] = -1
                            if "dissenting_citations" not in item:
                                item["dissenting_citations"] = []
                            if "concurring_opinion_citations" not in item:
                                item["concurring_opinion_citations"] = []
                            if "majority_opinion_citations" not in item:
                                item["majority_opinion_citations"] = []
                # If results is a dict, process it directly
                elif isinstance(results, dict):
                    if "relevance" in results and results["relevance"] == "":
                        results["relevance"] = -1
                    if "dissenting_citations" not in results:
                        results["dissenting_citations"] = []

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

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
        logger.error(f"Error in LLM job {job_id}: {str(e)}")
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
            message="Loading LLM results",
        )

        # Load the actual JSON file
        with open(llm_job["result_path"], "r", encoding="utf-8") as f:
            llm_results = json.load(f)

        # Resolve citations
        update_job_status(
            db,
            job_id,
            JobStatus.PROCESSING,
            progress=30.0,
            message="Resolving citations",
        )

        # Process and resolve citations
        resolved_citations = []
        resolution_errors = {}

        def is_valid_cluster_id(cluster_id):
            """Helper function to validate cluster IDs"""
            if (
                not cluster_id
                or cluster_id == "null"
                or cluster_id == "undefined"
                or str(cluster_id).strip() == ""
            ):
                return False
            return True

        def process_citation(citation_data):
            """Helper function to validate a citation and return a Citation object"""
            try:
                return Citation.model_validate(citation_data)
            except Exception as e:
                citation_text = (
                    citation_data["citation_text"]
                    if "citation_text" in citation_data
                    else "unknown"
                )
                logger.warning(f"Invalid citation: {citation_text}: {str(e)}")
                return None

        # Create a direct mapping of cluster_id to valid CitationAnalysis objects
        cluster_analyses = {}

        for cluster_id, results in llm_results.items():
            # Skip invalid cluster IDs
            if not is_valid_cluster_id(cluster_id):
                logger.warning(f"Invalid cluster ID: {cluster_id}, skipping")
                resolution_errors[str(cluster_id)] = {
                    "error": "Invalid cluster ID",
                    "input_data": results,
                }
                continue

            # Skip empty results
            if not results:
                logger.warning(f"Empty results for cluster {cluster_id}, skipping")
                continue

            try:
                # Process items and create CitationAnalysis objects directly
                items = results if isinstance(results, list) else [results]
                analyses = []

                for item in items:
                    if not isinstance(item, dict):
                        logger.warning(
                            f"Skipping non-dictionary item for cluster {cluster_id}"
                        )
                        continue

                    # Process citation lists directly with validation
                    majority_citations = []
                    if "majority_opinion_citations" in item and isinstance(
                        item["majority_opinion_citations"], list
                    ):
                        for citation in item["majority_opinion_citations"]:
                            citation_obj = process_citation(citation)
                            if citation_obj:
                                majority_citations.append(citation_obj)

                    concurring_citations = []
                    if "concurring_opinion_citations" in item and isinstance(
                        item["concurring_opinion_citations"], list
                    ):
                        for citation in item["concurring_opinion_citations"]:
                            citation_obj = process_citation(citation)
                            if citation_obj:
                                concurring_citations.append(citation_obj)

                    dissenting_citations = []
                    if "dissenting_citations" in item and isinstance(
                        item["dissenting_citations"], list
                    ):
                        for citation in item["dissenting_citations"]:
                            citation_obj = process_citation(citation)
                            if citation_obj:
                                dissenting_citations.append(citation_obj)

                    # Only create if we have required data
                    if (
                        "date" in item
                        or "brief_summary" in item
                        or majority_citations
                        or concurring_citations
                        or dissenting_citations
                    ):
                        try:
                            # Use direct dictionary access with fallbacks
                            date = (
                                item["date"]
                                if "date" in item
                                else datetime.now().date().isoformat()
                            )
                            brief_summary = (
                                item["brief_summary"]
                                if "brief_summary" in item
                                else f"Summary for cluster {cluster_id}"
                            )

                            # Create a validated CitationAnalysis object directly
                            analysis = CitationAnalysis(
                                date=date,
                                brief_summary=brief_summary,
                                majority_opinion_citations=majority_citations,
                                concurring_opinion_citations=concurring_citations,
                                dissenting_citations=dissenting_citations,
                            )
                            analyses.append(analysis)
                        except Exception as e:
                            logger.warning(
                                f"Error creating CitationAnalysis for cluster {cluster_id}: {str(e)}"
                            )

                # Store analyses if any were successfully created
                if analyses:
                    cluster_analyses[cluster_id] = analyses

            except Exception as e:
                logger.warning(f"Error processing cluster {cluster_id}: {str(e)}")
                resolution_errors[str(cluster_id)] = {
                    "error": str(e),
                    "input_data": results,
                }

        # Create CombinedResolvedCitationAnalysis objects from validated CitationAnalysis objects
        for cluster_id, analyses in cluster_analyses.items():
            try:
                # Create the combined analysis
                combined_analysis = CombinedResolvedCitationAnalysis.from_citations(
                    analyses, int(cluster_id)
                )
                if combined_analysis:
                    resolved_citations.append(combined_analysis)
            except Exception as e:
                logger.warning(
                    f"Error creating combined analysis for cluster {cluster_id}: {str(e)}"
                )
                resolution_errors[str(cluster_id)] = {
                    "error": f"Error in combined analysis: {str(e)}",
                    "analyses_count": len(analyses),
                }

        # Log summary information
        logger.info(
            f"Processed {len(llm_results)} clusters, created {len(resolved_citations)} valid resolved citations"
        )
        if resolution_errors:
            logger.warning(
                f"Encountered {len(resolution_errors)} errors during citation resolution"
            )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"resolved_citations_{timestamp}.json"
        output_path = os.path.join("/tmp", output_file)

        # Save only valid citation objects
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    citation.model_dump()
                    for citation in resolved_citations
                    if citation is not None
                ],
                f,
                indent=2,
                ensure_ascii=False,  # Preserve Unicode characters
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
        logger.error(f"Error in resolution job {job_id}: {str(e)}")
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
        update_job_status(
            db,
            job_id,
            JobStatus.PROCESSING,
            progress=10.0,
            message="Loading resolved citations",
        )

        # Load the actual JSON file
        with open(resolution_job["result_path"], "r") as f:
            resolved_citations_data = json.load(f)

            # DEBUG: Print the length and a sample of the raw data
            logger.info(
                f"Loaded {len(resolved_citations_data)} citation entries from JSON"
            )
            if len(resolved_citations_data) > 0:
                logger.info(f"First item type: {type(resolved_citations_data[0])}")
                logger.info(
                    f"First item keys: {resolved_citations_data[0].keys() if isinstance(resolved_citations_data[0], dict) else 'Not a dict'}"
                )

            # Safer validation with debugging
            resolved_citations = []
            for i, data in enumerate(resolved_citations_data):
                try:
                    # DEBUG: Print information about each item
                    if isinstance(data, list):
                        logger.error(f"Item {i} is a list, not an object: {data[:100]}")
                        continue

                    # Check if required fields exist
                    if not isinstance(data, dict):
                        logger.error(f"Item {i} is not a dictionary: {type(data)}")
                        continue

                    if "cluster_id" not in data:
                        logger.error(
                            f"Item {i} missing cluster_id: {list(data.keys())}"
                        )
                        continue

                    # Try validation
                    citation = CombinedResolvedCitationAnalysis.model_validate(data)
                    resolved_citations.append(citation)
                except Exception as e:
                    logger.error(f"Error validating item {i}: {str(e)}")
                    logger.error(f"Item data: {str(data)[:200]}...")

        # Log how many valid citations we found
        logger.info(
            f"Successfully validated {len(resolved_citations)} out of {len(resolved_citations_data)} citations"
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
            uri=os.getenv("NEO4J_URI", "host.docker.internal:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "courtlistener"),
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
        logger.error(f"Error in Neo4j job {job_id}: {str(e)}")
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
        if extraction_job["status"] != JobStatus.COMPLETED:
            logger.error(
                f"Extraction job {extraction_job_id} failed, aborting pipeline"
            )
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
            logger.error(
                f"Resolution job {resolution_job_id} failed, aborting pipeline"
            )
            return

        # Run Neo4j job
        run_neo4j_job(db, neo4j_session, neo4j_job_id, resolution_job_id)

        logger.info(f"Completed full pipeline")

    except Exception as e:
        logger.error(f"Error in full pipeline: {str(e)}")
