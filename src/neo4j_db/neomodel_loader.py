#!/usr/bin/env python3
"""
Neomodel-based Loader for Legal Citation Network

A clean synchronous implementation using neomodel's native functionality for loading and managing
legal citation data in Neo4j. This loader handles both basic citation relationships
and enriched citation metadata.
"""

import logging
import traceback
from datetime import datetime
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Any,
    Type,
    Union,
    cast,
)
import time
import os
from dotenv import load_dotenv
from neo4j_db import GraphDatabase
import uuid

from neomodel import config, install_all_labels, db
from .models import (
    Opinion,
    LegalDocument,
    CITATION_TYPE_TO_NODE_TYPE,
)
from src.llm_extraction.models import (
    CitationType,
    OpinionType,
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
            rel = citing_node.cites.connect(cited_node)

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

    def load_opinion_list(
        self, opinions_data: List[Dict[str, Any]], batch_size: Optional[int] = None
    ) -> List[Opinion]:
        """
        Load a list of opinions, creating or updating nodes as needed.

        Args:
            opinions_data: List of dictionaries containing opinion data
            batch_size: Optional batch size for processing

        Returns:
            List of created/updated Opinion nodes
        """
        if batch_size is None:
            batch_size = self.batch_size

        results = []
        failed_items = []

        # Process in batches
        for i in range(0, len(opinions_data), batch_size):
            try:
                batch = opinions_data[i : i + batch_size]
                processed_batch = []

                # Process each item in the batch
                for data in batch:
                    try:
                        # Handle date conversion if needed
                        if "date_filed" in data and isinstance(data["date_filed"], str):
                            data["date_filed"] = datetime.strptime(
                                data["date_filed"], "%Y-%m-%d"
                            ).date()

                        # Create citation_string if not present
                        if "citation_string" not in data and "cluster_id" in data:
                            data["citation_string"] = f"opinion:{data['cluster_id']}"

                        # Ensure required fields
                        if "cluster_id" not in data:
                            logger.error("Missing cluster_id for opinion")
                            failed_items.append(data)
                            continue

                        # Process and add to batch
                        processed_batch.append(data)
                    except Exception as e:
                        logger.error(f"Error processing opinion data: {str(e)}")
                        failed_items.append(data)

                # Create or update each opinion individually
                nodes = []
                for data in processed_batch:
                    try:
                        # Create or update using the model's class method
                        opinion = Opinion.get_or_create_from_cluster_id(
                            data["cluster_id"],
                            data.get(
                                "citation_string", f"opinion:{data['cluster_id']}"
                            ),
                            **{
                                k: v
                                for k, v in data.items()
                                if k not in ["cluster_id", "citation_string"]
                            },
                        )
                        nodes.append(opinion)
                    except Exception as e:
                        logger.error(
                            f"Error creating opinion {data.get('cluster_id')}: {str(e)}"
                        )
                        failed_items.append(data)

                results.extend(nodes)
                logger.info(
                    f"Processed batch of {len(nodes)} opinions (total: {len(results)}, failed: {len(failed_items)})"
                )
            except Exception as e:
                logger.error(f"Failed to process batch: {str(e)}")

        logger.info(
            f"Completed loading {len(results)} opinions, failed: {len(failed_items)}"
        )
        return results

    def _process_citation_pair(
        self,
        citing_id: int,
        cited_id: int,
        citation: CitationResolved,
        data_source: str,
    ):
        """
        Process a citation pair (citing_id, cited_id) and create a citation relationship.

        Args:
            citing_id: The cluster ID of the citing opinion.
            cited_id: The cluster ID of the cited opinion.
            citation: The CitationResolved object containing metadata for the relationship.
            data_source: The source of the citation data.

        Returns:
            Tuple containing (citing opinion, cited opinion, success flag)
        """
        # Check for valid IDs
        if not citing_id or not cited_id:
            logger.warning(f"Invalid citation pair: ({citing_id}, {cited_id})")
            return None, None, False

        try:
            # Find or create nodes for citing and cited opinions
            citing_opinion = Opinion.get_or_create_from_cluster_id(
                citing_id, citation_string=f"opinion:{citing_id}"
            )
            cited_opinion = Opinion.get_or_create_from_cluster_id(
                cited_id, citation_string=f"opinion:{cited_id}"
            )

            # Create the citation relationship with metadata from the citation object
            self.create_citation(
                citing_node=citing_opinion,
                cited_node=cited_opinion,
                citation=citation,
                data_source=data_source,
                opinion_section=None,
            )

            return citing_opinion, cited_opinion, True
        except Exception as e:
            logger.error(
                f"Error processing citation pair ({citing_id}, {cited_id}): {e}"
            )
            return None, None, False

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
        loaded_count = 0
        error_count = 0
        processed_count = 0
        skipped_non_opinion = 0

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
                        # Create or update the citing opinion with AI summary
                        citing_opinion = Opinion.get_or_create_from_cluster_id(
                            citation.cluster_id,
                            citation_string=f"opinion:{citation.cluster_id}",
                            ai_summary=citation.brief_summary,
                            date_filed=citation.date,
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
                                    resolved_citation.type == "judicial_opinion"
                                    and resolved_citation.primary_id
                                ):
                                    try:
                                        # Process the citation using the typed CitationResolved object
                                        citing, cited, success = (
                                            self._process_citation_pair(
                                                citation.cluster_id,
                                                int(resolved_citation.primary_id),
                                                resolved_citation,
                                                data_source,
                                            )
                                        )

                                        if success:
                                            processed_count += 1
                                        else:
                                            error_count += 1
                                    except Exception as e:
                                        logger.error(
                                            f"Error creating citation from {citation.cluster_id} to {resolved_citation.primary_id}: {e}"
                                        )
                                        error_count += 1
                                # For other document types, log but don't process yet
                                elif resolved_citation.type != "judicial_opinion":
                                    logger.info(
                                        f"Skipping non-opinion citation: {resolved_citation.type}"
                                    )
                                    skipped_non_opinion += 1
                                else:
                                    logger.warning(
                                        f"Missing primary_id for citation: {resolved_citation.citation_text}"
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


# Example usage (synchronous):
# def main():
#     NEO4J_URI = "localhost:7687"
#     NEO4J_USER = "neo4j"
#     NEO4J_PASSWORD = "courtlistener"
#     loader = NeomodelLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
#     opinions = [
#         {
#             "cluster_id": 1001,
#             "date_filed": "2020-01-01",
#             "case_name": "Smith v. Jones",
#             "court_id": 5,
#             "court_name": "Supreme Court",
#         }
#     ]
#     loader.load_opinions(opinions)
#     citation_pairs = [(1001, 1002), (1001, 1003)]
#     loader.load_citation_pairs(citation_pairs, source="courtlistener")
#
# if __name__ == "__main__":
#     main()
