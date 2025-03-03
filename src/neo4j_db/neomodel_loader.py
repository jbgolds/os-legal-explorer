#!/usr/bin/env python3
"""
Neomodel-based Loader for Legal Citation Network

A clean synchronous implementation using neomodel's native functionality for loading and managing
legal citation data in Neo4j. This loader handles both basic citation relationships
and enriched citation metadata.
"""

import logging
import traceback
from datetime import datetime, date
from typing import (
    List,
    Optional,
    Any,
)
import time
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase


from neomodel import config, install_all_labels, db
from .models import (
    Opinion,
    LegalDocument,
    CITATION_TYPE_TO_NODE_TYPE,
)
from src.llm_extraction.models import (
    CitationType,
    OpinionSection,
    CombinedResolvedCitationAnalysis,
    CitationResolved,
)

# Mapping from citation types to primary tables
CITATION_TYPE_TO_PRIMARY_TABLE = {
    CitationType.judicial_opinion: "opinion_cluster",
    CitationType.statutes_codes_regulations: "statutes",
    CitationType.constitution: "constitution",
    CitationType.administrative_agency_ruling: "administrative_rulings",
    CitationType.congressional_report: "congressional_reports",
    CitationType.external_submission: "external_submissions",
    CitationType.electronic_resource: "electronic_resources",
    CitationType.law_review: "law_reviews",
    CitationType.legal_dictionary: "legal_dictionaries",
    CitationType.other: "other_legal_documents",
}

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://host.docker.internal:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "courtlistener")

# Set connection URL
config.DATABASE_URL = (
    f"bolt://{NEO4J_USER}:{NEO4J_PASSWORD}@{NEO4J_URI.replace('bolt://', '')}"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Neo4j driver instance
_neo4j_driver = None


def get_neo4j_driver():
    """
    Get the Neo4j driver instance, creating it if it doesn't exist.

    Returns:
        Neo4j driver
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        # Extract credentials from connection URL if needed
        _neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _neo4j_driver


def get_primary_id(node: LegalDocument) -> Any:
    return node.primary_id


class NeomodelLoader:
    """
    Loader class for managing legal citation data using neomodel.
    Handles both basic citation relationships and enriched citation metadata.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        batch_size: int = 1000,
    ):
        """
        Initialize the loader with Neo4j connection details.

        Args:
            uri: Neo4j server URI, defaults to NEO4J_URI env var
            username: Neo4j username, defaults to NEO4J_USER env var
            password: Neo4j password, defaults to NEO4J_PASSWORD env var
            batch_size: Number of nodes to process in each batch
        """
        # Use parameters if provided, otherwise fall back to module-level defaults
        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.batch_size = batch_size

        # Ensure URI has protocol
        if self.uri and "://" not in self.uri:
            self.uri = f"bolt://{self.uri}"

        # Configure neomodel connection if not already set
        if not config.DATABASE_URL:
            raise ValueError("Neo4j connection not configured")
        try:
            install_all_labels()
        except Exception as e:
            logger.error(
                f"Schema installation warning (can usually be ignored): {str(e)}"
            )

    def create_citation(
        self,
        citing_node: LegalDocument,
        cited_node: LegalDocument,
        citation: CitationResolved,
        data_source: str = "unknown",
        opinion_section: Optional[str] = None,
    ) -> None:
        """
        Create a citation relationship between two nodes with metadata.

        Args:
            citing_node: The node that is citing
            cited_node: The node that is being cited
            citation: The CitationResolved object containing citation metadata
            data_source: Source of the citation data
            opinion_section: Section of the opinion where the citation appears (majority, concurring, dissenting)
        """
        try:
            # Create relationship
            connect_method = getattr(citing_node.cites, "connect")
            rel = connect_method(cited_node)

            # Set standard metadata fields
            rel.data_source = data_source

            # Set metadata from the citation object
            if citation.citation_text:
                rel.citation_text = citation.citation_text

            if citation.treatment:
                rel.treatment = citation.treatment

            if citation.reasoning:
                rel.reasoning = citation.reasoning

            if opinion_section:
                rel.opinion_section = opinion_section

            # Add resolution metadata
            rel.resolution_confidence = citation.resolution_confidence
            rel.resolution_method = citation.resolution_method

            # Add any additional fields that might be in the primary_id/primary_table
            if citation.primary_id:
                rel.primary_id = citation.primary_id

            if citation.primary_table:
                rel.primary_table = citation.primary_table

            # Save the relationship
            rel.save()

            # Log with more specific information
            citing_id = get_primary_id(citing_node)
            cited_id = get_primary_id(cited_node)
            logger.debug(
                f"Created citation: {type(citing_node).__name__}[{citing_id}] -> "
                f"{type(cited_node).__name__}[{cited_id}]"
            )

        except Exception as e:
            logger.error(f"Error creating citation: {e}")
            raise

    def load_enriched_citations(
        self, citations_data: List[CombinedResolvedCitationAnalysis], data_source: str
    ):
        """
        Load enriched citation data with resolution and analysis information.

        Args:
            citations_data: List of citation analysis objects with majority, concurring, and dissenting citations
            data_source: Source of the citation data
        """
        start_time = time.time()
        error_count = 0
        processed_count = 0

        for i, citation in enumerate(citations_data):
            try:
                if i % 100 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Processed {i}/{len(citations_data)} citations ({rate:.2f}/s)"
                    )

                # Skip if no cluster ID
                if not citation.cluster_id:
                    logger.warning(f"No cluster ID for citation: {citation}")
                    continue

                # Process the citation
                with db.transaction:
                    try:
                        # Convert string date to datetime.date object
                        date_filed = None
                        if citation.date:
                            try:
                                date_filed = datetime.strptime(
                                    citation.date, "%Y-%m-%d"
                                ).date()
                            except ValueError:
                                logger.warning(
                                    f"Invalid date format for citation {citation.cluster_id}: {citation.date}"
                                )
                                date_filed = date(
                                    1500, 1, 1
                                )  # year 1500 so we know it's wrong

                        # Create or update the citing opinion with AI summary
                        citing_opinion = Opinion.get_or_create_from_cluster_id(
                            citation.cluster_id,
                            citation_string=f"cluster-{citation.cluster_id}",  # TODO TEMP, WE NEED TO RESOLVE THIS
                            ai_summary=citation.brief_summary,
                            date_filed=date_filed,
                        )

                        # Process each citation type (majority, concurring, dissenting)
                        citation_lists = [
                            (
                                OpinionSection.majority.value,
                                citation.majority_opinion_citations,
                            ),
                            (
                                OpinionSection.concurring.value,
                                citation.concurring_opinion_citations,
                            ),
                            (
                                OpinionSection.dissenting.value,
                                citation.dissenting_citations,
                            ),
                        ]

                        for opinion_section, citation_list in citation_lists:
                            if not citation_list:
                                continue

                            for resolved_citation in citation_list:
                                # Check if citation has a primary_id for judicial opinions
                                if (
                                    resolved_citation.type
                                    == CitationType.judicial_opinion
                                ):
                                    try:
                                        # Process the citation using the already created citing_opinion
                                        cited_id = (
                                            int(resolved_citation.primary_id)
                                            if resolved_citation.primary_id
                                            else None
                                        )

                                        # Use citation_text instead of creating an ID-based citation string
                                        cited_opinion = Opinion.get_or_create_from_cluster_id(
                                            cited_id,
                                            citation_string=resolved_citation.citation_text,
                                        )

                                        # Update cited_opinion with additional data from resolved_citation
                                        for (
                                            dict_key,
                                            value,
                                        ) in resolved_citation.model_dump().items():
                                            dict_key = str(dict_key)
                                            if (
                                                hasattr(cited_opinion, dict_key)
                                                and value is not None
                                            ):
                                                setattr(cited_opinion, dict_key, value)
                                        cited_opinion.save()

                                        # Create the citation relationship directly
                                        self.create_citation(
                                            citing_node=citing_opinion,
                                            cited_node=cited_opinion,
                                            citation=resolved_citation,
                                            data_source=data_source,
                                            opinion_section=opinion_section,
                                        )
                                        processed_count += 1
                                    except Exception as e:
                                        logger.error(
                                            f"Error creating citation from {citation.cluster_id} to {resolved_citation.primary_id}: {e}"
                                        )
                                        error_count += 1
                                # For other document types, log but don't process yet
                                else:
                                    try:
                                        node_type = CITATION_TYPE_TO_NODE_TYPE[
                                            resolved_citation.type
                                        ]

                                        # Use citation_text as the citation_string
                                        cited_node = node_type.get_or_create(
                                            citation_string=resolved_citation.citation_text,
                                        )
                                        # update cited_node with data
                                        for (
                                            dict_key,
                                            value,
                                        ) in resolved_citation.model_dump().items():
                                            dict_key = str(dict_key)
                                            # Use setattr instead of direct attribute assignment
                                            # to dynamically set the attribute based on the key name
                                            if (
                                                hasattr(cited_node, dict_key)
                                                and value is not None
                                            ):
                                                setattr(cited_node, dict_key, value)

                                        cited_node.save()
                                        self.create_citation(
                                            citing_node=citing_opinion,
                                            cited_node=cited_node,
                                            citation=resolved_citation,
                                            data_source=data_source,
                                            opinion_section=opinion_section,
                                        )
                                        processed_count += 1
                                    except Exception as e:
                                        logger.error(
                                            f"Error creating citation from {citation.cluster_id} to {resolved_citation.primary_id}: {e}"
                                        )
                                        error_count += 1

                    except Exception as e:
                        logger.error(
                            f"Error processing opinion {citation.cluster_id}: {str(e)}"
                        )
                        traceback.print_exc()
                        error_count += 1

            except Exception as e:
                logger.error(
                    f"Error processing citation set {i+1}: {str(e)}\n{traceback.format_exc()}"
                )
                error_count += 1

        # Log final stats
        elapsed = time.time() - start_time
        rate = len(citations_data) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Loaded {processed_count} citations with {error_count} errors in {elapsed:.2f}s ({rate:.2f}/s)"
        )

        return processed_count
