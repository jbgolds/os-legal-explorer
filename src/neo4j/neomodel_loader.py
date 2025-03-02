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
from typing import List, Dict, Tuple, Optional, Any
import time
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import uuid  # Add this import at the top of the file with other imports

from neomodel import config, install_all_labels, db
from .models import Opinion
from src.llm_extraction.models import CombinedResolvedCitationAnalysis
import json

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


# Neo4j driver instance
_neo4j_driver = None


def get_neo4j_driver():
    """
    Get the Neo4j driver instance, creating it if it doesn't exist.

    Returns:
        GraphDatabase.driver: Neo4j driver
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        # Ensure URI has protocol
        uri = NEO4J_URI if "://" in NEO4J_URI else f"bolt://{NEO4J_URI}"
        _neo4j_driver = GraphDatabase.driver(uri, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _neo4j_driver


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    def create_or_update_opinion(self, data: Dict[str, Any]) -> Optional[Opinion]:
        """
        Create a new opinion node or update an existing one while preserving existing metadata.

        Note: citation_text is no longer a required property for the Opinion node.
        If citation_text is provided, it will be removed from the node data and can be used
        on the relationship instead.

        Args:
            data: Dictionary containing opinion data

        Returns:
            Opinion node instance or None if creation failed
        """

        if "cluster_id" not in data:
            raise ValueError("cluster_id is required for opinion creation/update")

        # Handle case where cluster_id is a list (this is a bug but we can handle it)
        if isinstance(data["cluster_id"], list):
            logger.warning(
                f"Received list for cluster_id, using first item: {data['cluster_id']}"
            )
            if data["cluster_id"] and len(data["cluster_id"]) > 0:
                data["cluster_id"] = data["cluster_id"][0]
            else:
                logger.error("Empty list for cluster_id, cannot process")
                return None

        # Handle case where cluster_id is not an integer or string
        if not isinstance(data["cluster_id"], (int, str)):
            try:
                data["cluster_id"] = int(data["cluster_id"])
                logger.warning(f"Converted cluster_id to int: {data['cluster_id']}")
            except (ValueError, TypeError):
                logger.error(
                    f"Invalid cluster_id type ({type(data['cluster_id'])}), cannot process: {data['cluster_id']}"
                )
                return None

        # Handle date conversion if needed
        if "date_filed" in data and isinstance(data["date_filed"], str):
            data["date_filed"] = datetime.strptime(
                data["date_filed"], "%Y-%m-%d"
            ).date()

        try:
            # First try to get existing opinion
            try:
                opinion = Opinion.nodes.get(cluster_id=data["cluster_id"])
                # Update only provided fields, preserving existing data
                for key, value in data.items():
                    if value is not None and key != "cluster_id":
                        setattr(opinion, key, value)
                opinion.save()
                logger.debug(f"Updated opinion with cluster_id: {opinion.cluster_id}")
                return opinion
            except Opinion.DoesNotExist:
                # Create new opinion if it doesn't exist
                if isinstance(data["cluster_id"], list):
                    if len(data["cluster_id"]) == 1:
                        data["cluster_id"] = data["cluster_id"][0]
                opinion = Opinion.create(
                    {
                        "cluster_id": data["cluster_id"],
                        **{
                            k: v
                            for k, v in data.items()
                            if k != "cluster_id" and v is not None
                        },
                    }
                )
                if len(opinion) == 1:
                    opinion = opinion[0]
                else:
                    logger.warning(
                        f"WARNING Created {len(opinion)} opinions with cluster_id: {data['cluster_id']}"
                    )
                # Handle the case where create returns a list

                logger.debug(
                    f"Created new opinion with cluster_id: {data['cluster_id']}"
                )
                return opinion

        except Exception as e:
            logger.error(
                f"Error creating/updating opinion {data.get('cluster_id')}: {str(e)}"
            )
            raise

    def create_citation(
        self,
        citing_opinion: Opinion,
        cited_opinion: Opinion,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create or update a citation relationship between two opinions using get_or_create.
        If relationship exists, updates metadata and maintains version history.

        Args:
            citing_opinion: The opinion doing the citing
            cited_opinion: The opinion being cited
            metadata: Optional metadata for the citation relationship
        """
        try:
            # Ensure we have metadata dict and timestamp
            metadata = metadata or {}
            current_time = datetime.now()
            metadata.setdefault("timestamp", current_time)
            metadata.setdefault("version", 1)
            metadata.setdefault("other_metadata_versions", "[]")

            # Use get_or_create to atomically get or create the relationship
            rel = citing_opinion.cites.relationship(cited_opinion)  # type: ignore
            if rel:
                # Relationship exists - update with new metadata while preserving history
                try:
                    other_metadata_versions = json.loads(rel.other_metadata_versions)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    other_metadata_versions = []

                # Track changes for audit trail
                changes = []
                for key, new_value in metadata.items():
                    if new_value is not None and hasattr(rel, key):
                        old_value = getattr(rel, key)
                        if old_value != new_value:
                            changes.append(
                                {
                                    "field": key,
                                    "old_value": old_value,
                                    "updated_at": current_time.isoformat(),
                                    "version": rel.version,
                                }
                            )

                if changes:
                    other_metadata_versions.extend(changes)
                    metadata["other_metadata_versions"] = json.dumps(
                        other_metadata_versions
                    )
                    metadata["version"] = rel.version + 1

                # Update relationship with new metadata
                citing_opinion.cites.replace(cited_opinion, metadata)  # type: ignore
                logger.debug(
                    f"Updated citation relationship: {citing_opinion.cluster_id}->"
                    f"{cited_opinion.cluster_id} with {len(changes)} field changes"
                )
            else:
                # Create new relationship
                citing_opinion.cites.connect(cited_opinion, metadata)  # type: ignore
                logger.debug(
                    f"Created new citation relationship: {citing_opinion.cluster_id}->"
                    f"{cited_opinion.cluster_id}"
                )

        except Exception as e:
            logger.error(
                f"Error creating/updating citation {citing_opinion.cluster_id}->"
                f"{cited_opinion.cluster_id}: {str(e)}"
            )
            raise

    def load_opinion_list(
        self, opinions_data: List[Dict[str, Any]], batch_size: Optional[int] = None
    ) -> List[Opinion]:
        """
        Load multiple opinions in batches using get_or_create's bulk creation capability.

        Args:
            opinions_data: List of opinion data dictionaries
            batch_size: Optional override for batch size

        Returns:
            List of created/updated Opinion instances
        """
        batch_size = batch_size or self.batch_size
        results = []
        failed_items = []

        for i in range(0, len(opinions_data), batch_size):
            batch = opinions_data[i : i + batch_size]
            try:
                # Convert date strings to datetime objects
                processed_batch = []
                for data in batch:
                    if "date_filed" in data and isinstance(data["date_filed"], str):
                        data["date_filed"] = datetime.strptime(
                            data["date_filed"], "%Y-%m-%d"
                        ).date()
                    # Filter out None values
                    properties = {k: v for k, v in data.items() if v is not None}
                    processed_batch.append(properties)

                # Bulk create/get all nodes in the batch
                nodes = Opinion.get_or_create(*processed_batch)
                results.extend(nodes)

                logger.info(
                    f"Processed batch of {len(nodes)} opinions (total: {len(results)}, failed: {len(failed_items)})"
                )
            except Exception as e:
                logger.error(f"Failed to process batch: {str(e)}")
                failed_items.extend(batch)

        if failed_items:
            logger.warning(
                f"Failed to process {len(failed_items)} opinions. Consider retrying these items separately."
            )
        return results

    def _process_citation_pair(
        self,
        citing_id: int,
        cited_id: int,
        metadata: dict,
        data_source: str,
    ):
        """Process a single citation pair."""
        citing = Opinion.nodes.get_or_none(cluster_id=citing_id)
        cited = Opinion.nodes.get_or_none(cluster_id=cited_id)

        if citing and cited:
            rel = citing.cites.relationship(cited)
            if not rel:
                citing.cites.connect(
                    cited,
                    {
                        "data_source": data_source,
                        **metadata,
                    },
                )

    def load_citation_pairs(
        self,
        citation_pairs: List[Tuple[int, int]],
        data_source: str,
    ):
        """
        Load citation pairs into Neo4j.

        Args:
            citation_pairs: List of (citing_id, cited_id) tuples
            data_source: Source identifier for the citations
        """
        for citing_id, cited_id in citation_pairs:
            self._process_citation_pair(citing_id, cited_id, {}, data_source)

    def load_enriched_citations(
        self, citations_data: List[CombinedResolvedCitationAnalysis], data_source: str
    ):
        """
        Load enriched citation data into Neo4j.

        Args:
            citations_data: List of citation data with metadata
            data_source: Source identifier for the citations
        """
        skipped_citations = 0
        skipped_clusters = []
        processed_count = 0
        total_time_start = time.time()

        # Process in batches for better performance and error resilience
        batch_size = max(
            1, min(len(citations_data), 10)
        )  # Process at most 10 at a time, but ensure at least 1

        for i in range(0, len(citations_data), batch_size):
            batch = citations_data[i : i + batch_size]
            batch_time_start = time.time()

            # Process each citation in the batch
            for citation in batch:
                # Verify citation is the expected type before processing
                if not hasattr(citation, "cluster_id"):
                    logger.error(
                        f"Invalid citation object (missing cluster_id): {type(citation)}"
                    )
                    skipped_clusters.append(
                        getattr(citation, "cluster_id", f"unknown-{i}")
                    )
                    continue

                # Process each citation in a separate transaction for isolation
                with db.transaction:
                    try:
                        # Create or update the citing opinion with AI summary
                        citing_opinion = self.create_or_update_opinion(
                            {
                                "cluster_id": citation.cluster_id,
                                "ai_summary": citation.brief_summary,
                                "date_filed": citation.date,
                            }
                        )

                        if citing_opinion is None:
                            logging.error(
                                f"Failed to create/update citing opinion {citation.cluster_id}, skipping"
                            )
                            skipped_clusters.append(citation.cluster_id)
                            continue

                        # Process citations by section
                        sections = {
                            "majority": citation.majority_opinion_citations,
                            "concurring": citation.concurring_opinion_citations,
                            "dissent": citation.dissenting_citations,
                        }

                        citation_count = 0
                        skipped_in_opinion = 0

                        for section_name, section_citations in sections.items():
                            for cite in section_citations:
                                # Skip invalid citations but continue processing others
                                try:
                                    # Handle different citation types differently
                                    if cite.type == "judicial_opinion":
                                        # For judicial opinions, skip if resolved_opinion_cluster is invalid
                                        if (
                                            not cite.resolved_opinion_cluster
                                            or cite.resolved_opinion_cluster < 0
                                        ):
                                            skipped_in_opinion += 1
                                            continue

                                        # Set node_id to the resolved cluster ID for judicial opinions
                                        node_id = cite.resolved_opinion_cluster
                                    else:
                                        # For non-judicial opinions (statutes, etc.), create a UUID if no resolution
                                        # Use the citation text to generate a deterministic UUID
                                        citation_hash = str(hash(cite.citation_text))
                                        node_id = (
                                            int(citation_hash)
                                            if citation_hash.isdigit()
                                            else int(
                                                uuid.uuid5(
                                                    uuid.NAMESPACE_DNS,
                                                    cite.citation_text,
                                                ).int
                                            )
                                            % 2**31
                                        )

                                    # Instead of skipping citations with missing citation_text,
                                    # provide a default value and log a warning
                                    citation_text = cite.citation_text
                                    if not citation_text:
                                        citation_text = f"Citation from {citation.cluster_id} to {node_id}"
                                        logging.warning(
                                            f"Using default citation_text for citation from {citation.cluster_id} to {node_id} in {section_name} section"
                                        )

                                    # Create or update the cited opinion/statute
                                    # For non-judicial opinions, add additional type metadata
                                    node_props = {
                                        "cluster_id": node_id,
                                        "type": cite.type,
                                    }

                                    # Add citation text as a property for non-judicial opinions to aid identification
                                    if cite.type != "judicial_opinion":
                                        node_props["citation_text"] = cite.citation_text

                                    cited_node = self.create_or_update_opinion(
                                        node_props
                                    )

                                    if cited_node is None:
                                        logging.warning(
                                            f"Failed to create/update cited node {node_id}, skipping"
                                        )
                                        skipped_in_opinion += 1
                                        continue

                                    # Create the citation relationship with metadata
                                    metadata = {
                                        "data_source": data_source,
                                        "opinion_section": section_name,
                                        "treatment": cite.treatment,
                                        "relevance": cite.relevance,
                                        "reasoning": cite.reasoning,
                                        "citation_text": citation_text,  # Use the variable with possible default value
                                        "page_number": cite.page_number,
                                    }

                                    # Filter out None values
                                    metadata = {
                                        k: v
                                        for k, v in metadata.items()
                                        if v is not None
                                    }

                                    # Check if relationship already exists
                                    citing_opinion.cites.connect(
                                        cited_node, metadata
                                    )  # type: ignore
                                    logger.debug(
                                        f"Created new citation relationship: {citing_opinion.cluster_id}->"
                                        f"{cited_node.cluster_id}"
                                    )

                                    citation_count += 1

                                except Exception as e:
                                    logging.error(
                                        f"Error processing citation from {citation.cluster_id} to {getattr(cite, 'resolved_opinion_cluster', 'unknown')} in {section_name} section: {str(e)}\n{traceback.format_exc()}"
                                    )
                                    skipped_in_opinion += 1

                        # Update counts
                        skipped_citations += skipped_in_opinion
                        processed_count += 1

                        logging.info(
                            f"Processed opinion: {citation.cluster_id} with {citation_count} citations ({skipped_in_opinion} skipped)"
                        )

                    except Exception as e:
                        logging.error(
                            f"Error processing opinion {citation.cluster_id}, skipping: {str(e)}; {traceback.format_exc()}"
                        )
                        skipped_clusters.append(citation.cluster_id)

            # Log batch statistics
            batch_time = time.time() - batch_time_start
            logging.info(
                f"Processed batch of {len(batch)} opinions in {batch_time:.2f} seconds"
            )

        # Log summary statistics
        total_time = time.time() - total_time_start
        logging.info(
            f"Citation loading complete: {processed_count}/{len(citations_data)} analyses processed, "
            f"{len(skipped_clusters)} clusters skipped, {skipped_citations} citations skipped, "
            f"total time: {total_time:.2f} seconds"
        )

        if skipped_clusters:
            logging.info(f"Clusters to consider for reprocessing: {skipped_clusters}")

        return {
            "total_analyses": len(citations_data),
            "processed": processed_count,
            "skipped_clusters": skipped_clusters,
            "skipped_citations": skipped_citations,
            "processing_time": total_time,
        }


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
