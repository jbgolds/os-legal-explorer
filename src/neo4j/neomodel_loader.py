#!/usr/bin/env python3
"""
Neomodel-based Loader for Legal Citation Network

A clean synchronous implementation using neomodel's native functionality for loading and managing
legal citation data in Neo4j. This loader handles both basic citation relationships
and enriched citation metadata.
"""

import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import time
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

from neomodel import config, install_all_labels
from .models import Opinion
from src.llm_extraction.models import CombinedResolvedCitationAnalysis
import json

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "courtlistener")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "courtlistener")

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
        _neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE
        )
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
        uri: str,
        username: str,
        password: str,
        database: str = "courtlistener",
        batch_size: int = 1000,
    ):
        """
        Initialize the loader with connection details and settings.

        Args:
            uri: Neo4j server URI
            username: Neo4j username
            password: Neo4j password
            database: Name of the Neo4j database to use
            batch_size: Default batch size for operations
        """
        self.uri = uri
        self.database = database
        self.batch_size = batch_size

        # Configure neomodel connection
        config.DATABASE_URL = f"bolt://{username}:{password}@{uri}/{database}"

        # Install constraints and indexes defined in the models
        self.setup_schema()

    def setup_schema(self) -> None:
        """
        Install all constraints and indexes defined in the models.
        This is handled automatically by neomodel based on model definitions.
        """
        try:
            install_all_labels()
            logger.info("Successfully installed schema constraints and indexes")
        except Exception as e:
            logger.error(f"Error setting up schema: {str(e)}")
            raise

    def create_or_update_opinion(self, data: Dict[str, Any]) -> Opinion:
        """
        Create a new opinion node or update an existing one while preserving existing metadata.

        Args:
            data: Dictionary containing opinion data

        Returns:
            Opinion node instance
        """
        if "cluster_id" not in data:
            raise ValueError("cluster_id is required for opinion creation/update")

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
                logger.debug(
                    f"Created new opinion with cluster_id: {opinion.cluster_id}"
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
            rel = citing_opinion.cites.relationship(cited_opinion)
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
                citing_opinion.cites.replace(cited_opinion, metadata)
                logger.debug(
                    f"Updated citation relationship: {citing_opinion.cluster_id}->"
                    f"{cited_opinion.cluster_id} with {len(changes)} field changes"
                )
            else:
                # Create new relationship
                citing_opinion.cites.connect(cited_opinion, metadata)
                logger.debug(
                    f"Created new citation relationship: {citing_opinion.cluster_id}->"
                    f"{cited_opinion.cluster_id}"
                )

            # Update citation count if available
            try:
                if hasattr(cited_opinion, "increment_citation_count"):
                    cited_opinion.increment_citation_count()
            except Exception as count_error:
                logger.warning(
                    f"Failed to increment citation count for opinion {cited_opinion.cluster_id}: {str(count_error)}"
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
        for citation in citations_data:
            metadata = {
                "treatment": citation.treatment,
                "relevance": citation.relevance,
                "reasoning": citation.reasoning,
                "citation_text": citation.citation_text,
                "data_source": data_source,
            }

            # Create or update the citing opinion with AI summary
            citing_opinion = self.create_or_update_opinion(
                {
                    "cluster_id": citation.cluster_id,
                    "ai_summary": citation.brief_summary,
                    "date_filed": citation.date,
                }
            )

            # Make this a dict with the section name as the key and the citations as the value
            sections = {
                "majority": citation.majority_opinion_citations,
                "concurrent": citation.concurrent_opinion_citations,
                "dissent": citation.dissenting_citations,
            }

            for section_name, citations in sections.items():
                for citation in citations:
                    if citation.resolved_opinion_cluster:
                        # Create or update the cited opinion
                        cited_opinion = self.create_or_update_opinion(
                            {
                                "cluster_id": citation.resolved_opinion_cluster,
                                "type": citation.type,
                            }
                        )

                        # Create the enriched citation relationship
                        metadata = {
                            "data_source": data_source,
                            "opinion_section": section_name,
                            "treatment": citation.treatment,
                            "relevance": citation.relevance,
                            "reasoning": citation.reasoning,
                            "citation_text": citation.citation_text,
                            "page_number": citation.page_number,
                        }

                        # Filter out None values
                        metadata = {k: v for k, v in metadata.items() if v is not None}

                        self.create_citation(citing_opinion, cited_opinion, metadata)


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
