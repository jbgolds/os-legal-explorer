#!/usr/bin/env python3
"""
Neomodel-based Loader for Legal Citation Network

A clean synchronous implementation using neomodel's native functionality for loading and managing
legal citation data in Neo4j. This loader handles both basic citation relationships
and enriched citation metadata.



TODO: 
- Add support for bulk loading from csv/faster.
"""
import logging
import os
import time
import traceback
from datetime import date, datetime
from typing import Any, List, Optional

from dotenv import load_dotenv
from neomodel import config, adb, db, install_all_labels
from tqdm import tqdm

from src.llm_extraction.models import (CitationResolved, CitationType,
                                       CombinedResolvedCitationAnalysis,
                                       OpinionSection)
from src.neo4j_db.models import (CITATION_TYPE_TO_NODE_TYPE, CitesRel,
                                 LegalDocument, Opinion)

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

# Filter to mute CartesianProduct notifications
class CartesianProductFilter(logging.Filter):
    def filter(self, record):
        if "CartesianProduct" in record.getMessage():
            return False
        return True

logging.getLogger("neo4j.notifications").addFilter(CartesianProductFilter())

def check_schema_exists():
    """
    Check if schema (node labels, indexes, constraints) already exists in the database.

    Returns:
        bool: True if any schema elements exist, False otherwise
    """
    try:
        # Check if any of our expected node labels exist
        query = """
        CALL db.labels() YIELD label
        RETURN count(label) > 0 AS has_labels
        """
        results, _ = db.cypher_query(query, {})
        has_labels = results and results[0][0]
        
        if not has_labels:
            # Check if any indexes exist
            query = """
            CALL db.indexes() 
            RETURN count(*) > 0 AS has_indexes
            """
            try:
                results, _ = db.cypher_query(query, {})
                has_indexes = results and results[0][0]
            except Exception as e:
                logger.debug(f"Error checking indexes with adb.indexes(): {str(e)}")
                # Fallback for Neo4j 4.0+
                try:
                    query = """
                    SHOW INDEXES
                    RETURN count(*) > 0 AS has_indexes
                    """
                    results, _ = db.cypher_query(query, {})
                    has_indexes = results and results[0][0]
                except Exception as e:
                    logger.debug(f"Error checking indexes with SHOW INDEXES: {str(e)}")
                    has_indexes = False
            
            return has_labels or has_indexes
        return True
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {str(e)}")
        # If we can't connect, assume no schema
        return False

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
    ):
        """
        Initialize the loader with Neo4j connection details.

        Args:
            uri: Neo4j server URI, defaults to NEO4J_URI env var
            username: Neo4j username, defaults to NEO4J_USER env var
            password: Neo4j password, defaults to NEO4J_PASSWORD env var
        """
        # Use parameters if provided, otherwise fall back to module-level defaults
        self.url = url or DB_NEO4J_URL
        self.username = username or NEO4J_USER
        self.password = password or NEO4J_PASSWORD

        # Ensure URI has protocol
        if self.url and "://" not in self.url:
            self.url = f"bolt://{self.url}"

        # Configure neomodel connection if not already set
        if not config.DATABASE_URL:
            config.DATABASE_URL = f"bolt://{self.username}:{self.password}@{self.url}"

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

    @adb.read_transaction
    async def find_existing_relationship(self, citing_node: LegalDocument, cited_node: LegalDocument, citation_text: str, page_number: Optional[int] = None) -> Optional[CitesRel]:
        """
        Find an existing relationship between two nodes with matching citation_text and page_number.
        """
        params = {
            "start_id": citing_node.element_id,
            "end_id": cited_node.element_id,
            "citation_text": citation_text
        }
    
        # Build query based on whether page_number is provided
        if page_number is not None:
            query = (
                "MATCH (a)-[r:CITES]->(b) "
                "WHERE elementId(a) = $start_id AND elementId(b) = $end_id "
                "AND r.citation_text = $citation_text "
                "AND r.page_number = $page_number "
                "RETURN r"
            )
            params["page_number"] = page_number
        else:
            query = (
                "MATCH (a)-[r:CITES]->(b) "
                "WHERE elementId(a) = $start_id AND elementId(b) = $end_id "
                "AND r.citation_text = $citation_text "
                "AND NOT exists(r.page_number) "
                "RETURN r"
            )
        
        results, _ = await adb.cypher_query(query, params, resolve_objects=True)
        if results and results[0]:
            if len(results) > 1:
                logger.warning(f"Found multiple relationships with citation_text {citation_text} and page_number {page_number}")
            return results[0][0]
        return None

    @adb.transaction
    async def create_citation(
        self,
        citing_node: LegalDocument,
        cited_node: LegalDocument,
        citation: CitationResolved,
        data_source: str = "unknown",
        opinion_section: Optional[str] = None,
    ) -> None:
        """
        create a citation relationship between two nodes with metadata.

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
            
            # Get citation_text from the citation object or properties
            citation_text = getattr(citation, "citation_text", None) or properties.get("citation_text")
            page_number = getattr(citation, "page_number", None) or properties.get("page_number")

            # Check if we have a relationship with this citation text and page number already
            existing_rel = None

            # Build query based on whether page_number is available
            existing_rel = await self.find_existing_relationship(
                citing_node, 
                cited_node, 
                citation_text or "", 
                int(page_number) if page_number else None
            )

            if existing_rel:
                logger.debug(
                    f"Relationship already exists between {citing_node} and {cited_node} with citation_text '{citation_text}'"
                )

                await existing_rel.update_history(properties)
                return

            # Create relationship by passing a dictionary of properties, per neomodel docs
            rel = await citing_node.cites.connect(cited_node, properties)  # type: ignore

            # Logging relationship properties after connecting
            logger.debug(f"Created relationship with data_source='{rel.data_source}'")
            logger.debug(
                f"CitesRel properties after connecting: data_source={rel.data_source}, citation_text={getattr(rel, 'citation_text', None)}, treatment={getattr(rel, 'treatment', None)}"
            )

            # Save the relationship
            try:
                await rel.save()
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
                citing_id = citing_node.primary_id if citing_node else "None"
                cited_id = cited_node.primary_id if cited_node else "None"
                citation_text_snippet = citation_text[0:10] if citation_text else "None"
                logger.error(
                    f"Error creating citation from {citing_id} to {cited_id} with citation_text {citation_text_snippet} and page_number {page_number}: {e}, \n {traceback.format_exc()}"
                )
            except Exception:
                pass
            raise
    
    @adb.transaction
    async def load_enriched_citations(
        self, citations_data: List[CombinedResolvedCitationAnalysis], data_source: str
    ) -> None:
        """
        Load enriched citation data with resolution and analysis information.

        Args:
            citations_data: List of citation analysis objects with majority, concurring, and dissenting citations
            data_source: Source of the citation data
        """
        error_count = 0
        processed_count = 0

        for citation in tqdm(citations_data, desc="Loading opinion citations", unit="opinion"):
                # Use neomodel's transaction context manager              
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
                citing_opinion = await Opinion.get_or_create_from_cluster_id(
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
                                cited_opinion = await Opinion.get_or_create_from_cluster_id(
                                    cluster_id=cited_id,
                                    citation_string=resolved_citation.citation_text, 
                                )

                                # Create the citation relationship directly
                                await self.create_citation(
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
                                citation_text_snippet = resolved_citation.citation_text[0:10] if resolved_citation.citation_text else "None"
                                logger.error(
                                    f"Error creating citation from {citing_id} to {cited_id} with citation_text {citation_text_snippet}: {e} \n {traceback.format_exc()}"
                                )
                                error_count += 1
                        # For other document types, log but don't process yet
                        else:
                            try:
                                node_type = CITATION_TYPE_TO_NODE_TYPE[
                                    resolved_citation.type
                                ]
                                table_name = CITATION_TYPE_TO_PRIMARY_TABLE[
                                    resolved_citation.type
                                ]

                                # Use citation_text as the citation_string

                                # Using built in get_or_create method

                                cited_node = await node_type.nodes.first_or_none(citation_string=resolved_citation.citation_text)
                                if not cited_node:
                                    cited_node = node_type(
                                        citation_string=resolved_citation.citation_text,
                                        primary_table=table_name,
                                    )
                                    await cited_node.save()

                                await self.create_citation(
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
                                citation_text_snippet = resolved_citation.citation_text[0:10] if resolved_citation.citation_text else "None"
                                logger.error(
                                    f"Error creating citation from {citing_id} to {cited_id} with citation_text {citation_text_snippet}: {e} \n {traceback.format_exc()}"
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

           

        logger.info(f"Loaded {processed_count} citations with {error_count} errors")

        return

    async def get_case_from_neo4j(self, cluster_id: str) -> Optional[Opinion]:
        """
        Get a case from Neo4j based on its cluster ID.

        Args:
            cluster_id: The cluster ID of the case to retrieve

        Returns:
            Optional[Opinion]: The case if found, otherwise None
        """
        try:
            return await Opinion.nodes.first_or_none(primary_id=cluster_id)
        except Exception as e:
            logger.error(f"Error getting case from Neo4j: {e}")
            return None



neomodel_loader = NeomodelLoader(         url=os.environ["DB_NEO4J_URL"],
            username=os.environ["DB_NEO4J_USER"],
            password=os.environ["DB_NEO4J_PASSWORD"],)