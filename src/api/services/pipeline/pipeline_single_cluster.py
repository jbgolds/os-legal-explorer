"""
Process a single cluster through the citation pipeline.

This module provides functionality to process a single cluster ID through the
citation extraction, LLM processing, resolution, and Neo4j loading pipeline.
"""

import logging
from typing import Any, Dict

from sqlalchemy.orm import Session

from src.api.services.pipeline import pipeline_service
from src.api.services.pipeline.pipeline_model import ExtractionConfig, JobType

logger = logging.getLogger(__name__)


def process_single_cluster(
    db: Session, cluster_id: int
) -> Dict[str, Any]:
    """
    Process a single cluster through the pipeline.

    Args:
        db: Database session
        neo4j_session: Neo4j session
        cluster_id: Cluster ID to process

    Returns:
        Dictionary with job IDs for each step
    """
    try:
        logger.info(f"Processing cluster {cluster_id}")

        # Create a custom extraction config for this cluster
        config = ExtractionConfig(
            court_id=None,
            start_date=None,
            end_date=None,
            limit=None,
            offset=0,
            include_text=True,
            include_metadata=True,
            single_cluster_id=cluster_id,
        )

        # Create jobs for each step
        extraction_job_id = pipeline_service.create_job(
            db, JobType.EXTRACT, config.model_dump()
        )
        llm_job_id = pipeline_service.create_job(
            db, JobType.LLM_PROCESS, {"extraction_job_id": extraction_job_id}
        )
        resolution_job_id = pipeline_service.create_job(
            db, JobType.CITATION_RESOLUTION, {"llm_job_id": llm_job_id}
        )
        neo4j_job_id = pipeline_service.create_job(
            db, JobType.NEO4J_LOAD, {"resolution_job_id": resolution_job_id}
        )

        # Run the pipeline
        pipeline_service.run_full_pipeline(
            db,
            extraction_job_id,
            llm_job_id,
            resolution_job_id,
            neo4j_job_id,
            config,
        )

        return {
            "extraction_job_id": extraction_job_id,
            "llm_job_id": llm_job_id,
            "resolution_job_id": resolution_job_id,
            "neo4j_job_id": neo4j_job_id,
            "cluster_id": cluster_id,
        }

    except Exception as e:
        logger.error(f"Error processing cluster {cluster_id}: {e}")
        raise
