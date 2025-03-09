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
from src.neo4j_db.models import CitesRel, LegalDocument, Opinion

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

# Neo4j Configuration using .env variables
DB_NEO4J_URL = os.environ["DB_NEO4J_URL"]
NEO4J_USER = os.environ["DB_NEO4J_USER"]
NEO4J_PASSWORD = os.environ["DB_NEO4J_PASSWORD"] 



# Set connection URL - fixing the format to ensure proper URL encoding
config.DATABASE_URL = f"bolt://{NEO4J_USER}:{NEO4J_PASSWORD}@{DB_NEO4J_URL}"


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
    Implements retry logic with exponential backoff to handle Neo4j startup delays.

    Returns:
        Neo4j driver
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        max_retries = 5
        retry_count = 0
        base_delay = 2  # Start with 2 seconds delay
        
        while retry_count < max_retries:
            try:
                if retry_count == 0:
                    time.sleep(15)
                # Use the GraphDatabase.driver method with separate auth parameter
                _neo4j_driver = GraphDatabase.driver(
                    uri=f"bolt://{DB_NEO4J_URL}", 
                    auth=(NEO4J_USER, NEO4J_PASSWORD)
                )
                # Verify connection works immediately to catch auth errors
                _neo4j_driver.verify_connectivity()
                logger.info(f"Successfully connected to Neo4j at bolt://{DB_NEO4J_URL} with user {NEO4J_USER}")
                return _neo4j_driver
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to create Neo4j driver after {max_retries} attempts: {e}")
                    raise e
                
                # Calculate delay with exponential backoff (2s, 4s, 8s, 16s, 32s)
                delay = base_delay * (2 ** (retry_count - 1))
                logger.warning(f"Neo4j connection attempt {retry_count} failed. Retrying in {delay} seconds. Error: {e}")
                time.sleep(delay)
    
    return _neo4j_driver


def check_schema_exists():
    """
    Check if schema (node labels, indexes, constraints) already exists in the database.

    Returns:
        bool: True if any schema elements exist, False otherwise
    """
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            # Check if any of our expected node labels exist
            try:
                result = session.run(
                    """
                    CALL db.labels() YIELD label
                    RETURN count(label) > 0 AS has_labels
                    """
                )
                record = result.single()
                has_labels = record and record.get("has_labels", False)
            except Exception as e:
                logger.debug(f"Error checking labels: {str(e)}")
                has_labels = False

            # Check if any indexes exist - using db.indexes() instead of SHOW INDEXES for compatibility
            has_indexes = False
            try:
                result = session.run(
                    """
                    CALL db.indexes() 
                    RETURN count(*) > 0 AS has_indexes
                    """
                )
                record = result.single()
                has_indexes = record and record.get("has_indexes", False)
            except Exception as e:
                logger.debug(f"Error checking indexes with db.indexes(): {str(e)}")
                # Fallback for Neo4j 4.0+
                try:
                    result = session.run(
                        """
                        SHOW INDEXES
                        RETURN count(*) > 0 AS has_indexes
                        """
                    )
                    record = result.single()
                    has_indexes = record and record.get("has_indexes", False)
                except Exception as e:
                    logger.debug(f"Error checking indexes with SHOW INDEXES: {str(e)}")
                    # If both fail, assume no indexes
                    has_indexes = False

            return has_labels or has_indexes
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {str(e)}")
        # If we can't connect, assume no schema
        return False


def get_primary_id(node: LegalDocument) -> Any:
    return node.primary_id


class NeomodelLoader:
    """
    Loader class for managing legal citation data using neomodel.
    Handles both basic citation relationships and enriched citation metadata.
    """

    def __init__(
        self,
        url: Optional[str] = None,
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
        self.url = url or DB_NEO4J_URL
        self.username = username or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.batch_size = batch_size

        # Ensure URI has protocol
        if self.url and "://" not in self.url:
            self.url = f"bolt://{self.url}"

        # Configure neomodel connection if not already set
        if not config.DATABASE_URL:
            raise ValueError("Neo4j connection not configured")

        # Only install labels if no schema exists yet
        if not check_schema_exists():
            logger.info("No existing schema found, installing labels...")
            try:
                install_all_labels()
                logger.info("Labels successfully installed")
            except Exception as e:
                logger.error(f"Error installing labels: {str(e)}")
        else:
            logger.info("Schema already exists, skipping label installation")

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
            # Ensure data_source is set
            if not data_source:
                data_source = "unknown"
                logger.warning("data_source was not provided or empty, using 'unknown'")

            # Define properties dictionary with required data_source
            properties = {"data_source": str(data_source)}

            # Add properties from citation object that exist in CitesRel
            citation_dict = citation.model_dump()
            rel_properties = list(CitesRel.defined_properties().keys())
            for prop in rel_properties:
                if prop in citation_dict and citation_dict[prop] is not None:
                    properties[prop] = citation_dict[prop]

            # Add opinion section if provided
            if opinion_section:
                properties["opinion_section"] = opinion_section

            # see if the relationship already exists and update metadata with previous version,
            # and set the version to the next number
            # and update the fields

            if existing_rel := citing_node.cites.relationship(cited_node):  # type: ignore
                logger.debug(
                    f"Relationship already exists between {citing_node} and {cited_node}"
                )

                # Use the __properties__ property which returns a dictionary of the
                # actual property values, not the property definitions
                current_properties = existing_rel.__properties__.copy()

                # Remove other_metadata_versions from the properties to avoid circular references
                if "other_metadata_versions" in current_properties:
                    current_properties.pop("other_metadata_versions")

                # Convert timestamp to datetime object before storing in version history
                for time_obj in ["created_at", "updated_at"]:
                    if time_obj in current_properties:
                        if isinstance(current_properties[time_obj], (int, float)):
                            current_properties[time_obj] = datetime.fromtimestamp(
                                current_properties[time_obj]
                            )
                        # Format datetime as string for JSON serialization
                        current_properties[time_obj] = current_properties[
                            time_obj
                        ].strftime("%Y-%m-%d %H:%M:%S")

                # Append the current properties to other_metadata_versions
                current_versions = existing_rel.other_metadata_versions or []
                current_versions.append(current_properties)
                existing_rel.other_metadata_versions = current_versions

                # Increment the version number
                existing_rel.version = existing_rel.version + 1

                # Update properties on the existing relationship
                for key, value in properties.items():
                    setattr(existing_rel, key, value)

                existing_rel.save()
                return

            # Create relationship by passing a dictionary of properties, per neomodel docs
            rel = citing_node.cites.connect(cited_node, properties)  # type: ignore

            # Logging relationship properties after connecting
            logger.debug(f"Created relationship with data_source='{rel.data_source}'")
            logger.debug(
                f"CitesRel properties after connecting: data_source={rel.data_source}, citation_text={getattr(rel, 'citation_text', None)}, treatment={getattr(rel, 'treatment', None)}"
            )

            # Save the relationship
            try:
                rel.save()
                # Log with more specific information
                citing_id = get_primary_id(citing_node)
                cited_id = get_primary_id(cited_node)
                logger.debug(
                    f"Created citation: {type(citing_node).__name__}[{citing_id}] -> "
                    f"{type(cited_node).__name__}[{cited_id}]"
                )
            except Exception as save_error:
                logger.error(f"Error saving citation relationship: {save_error}")
                logger.error(traceback.format_exc())
                logger.debug(
                    f"CitesRel properties: data_source={rel.data_source}, citation_text={getattr(rel, 'citation_text', None)}"
                )
                raise

        except Exception as e:
            logger.error(f"Error creating citation: {e}")
            logger.error(traceback.format_exc())
            # Attempt to log more specific information for debugging
            try:
                citing_id = get_primary_id(citing_node) if citing_node else "None"
                cited_id = get_primary_id(cited_node) if cited_node else "None"
                logger.error(
                    f"Error creating citation from {citing_id} to {cited_id}: {e}"
                )
            except Exception:
                pass
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
                                        # Make sure we're only using variables that are definitely defined
                                        citing_id = (
                                            citation.cluster_id
                                            if hasattr(citation, "cluster_id")
                                            else "unknown"
                                        )
                                        cited_id = (
                                            resolved_citation.primary_id
                                            if hasattr(resolved_citation, "primary_id")
                                            else "None"
                                        )
                                        logger.error(
                                            f"Error creating citation from {citing_id} to {cited_id}: {e}"
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
                                        # Make sure we're only using variables that are definitely defined
                                        citing_id = (
                                            citation.cluster_id
                                            if hasattr(citation, "cluster_id")
                                            else "unknown"
                                        )
                                        cited_id = (
                                            resolved_citation.primary_id
                                            if hasattr(resolved_citation, "primary_id")
                                            else "None"
                                        )
                                        logger.error(
                                            f"Error creating citation from {citing_id} to {cited_id}: {e}"
                                        )
                                        error_count += 1

                    except Exception as e:
                        # Make sure we're only using variables that are definitely defined
                        cluster_id = (
                            citation.cluster_id
                            if hasattr(citation, "cluster_id")
                            else "unknown"
                        )
                        logger.error(f"Error processing opinion {cluster_id}: {str(e)}")
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

    def get_case_from_neo4j(self, cluster_id: str) -> Optional[Opinion]:
        """
        Get a case from Neo4j based on its cluster ID.

        Args:
            cluster_id: The cluster ID of the case to retrieve

        Returns:
            Optional[Opinion]: The case if found, otherwise None
        """
        try:
            with db.transaction:
                return Opinion.nodes.first_or_none(primary_id=cluster_id)
        except Exception as e:
            logger.error(f"Error getting case from Neo4j: {e}")
            return None
