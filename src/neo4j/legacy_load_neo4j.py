"""
Neo4j Citation Network Loader

This module provides functionality for loading legal citation networks into Neo4j, supporting both basic
citation relationships and rich metadata from various data sources.

Loading Strategies:
1. Initial Bulk Load (Recommended for new databases):
   - Use neo4j-admin import tool (see neo4j_import/import_opinions.sh)
   - Requires database to be stopped
   - 10-100x faster than Cypher imports
   - Creates all Opinion nodes with proper indexes
   
2. Database Management:
   - For resetting/clearing database, use neo4j-admin commands:
     a) Stop Neo4j:    docker compose stop neo4j
     b) Drop database: docker exec neo4j cypher-shell "DROP DATABASE courtlistener IF EXISTS;"
     c) Create fresh:  docker exec neo4j cypher-shell "CREATE DATABASE courtlistener;"
     d) Start Neo4j:   docker compose up -d
   - These commands are more efficient than Cypher-based clearing
   - See Makefile for automation of these steps
   
3. Incremental Updates (This module):
   - Use this Python loader for:
     a) Adding new opinions after initial bulk load
     b) Managing citation relationships
     c) Updating metadata
     d) Adding rich citation data
   - Maintains data consistency
   - Supports versioning
   - Handles complex metadata

Key Design Decisions:
1. Opinion Node Structure:
   - Core required properties match the source PostgreSQL database:
     - cluster_id (INTEGER): Maps to 'id' in source database
     - date_filed (DATE): When the opinion was filed
     - case_name (STRING): Name of the case
     - docket_id (INTEGER): Foreign key to docket table
     - docket_number (STRING): Text representation of docket
     - court_id (INTEGER): Foreign key to court table
     - court_name (STRING): Text representation of court
   - Optional properties:
     - scdb_votes_majority (INTEGER): Supreme Court Database majority votes
     - scdb_votes_minority (INTEGER): Supreme Court Database minority votes

2. Citation Relationships:
   - Each relationship includes:
     - Required: source (STRING) to track data origin (e.g., "westlaw", "lexis")
     - Required: version (INTEGER) for tracking multiple citations
     - Optional: treatment, relevance, reasoning, etc.
   - Versioning Strategy:
     - When exact metadata matches exist, increment version number
     - When metadata differs, create new relationship with version 1
     - Timestamps track creation/update times

3. Batch Processing:
   - Operations are batched for performance (default 50,000 items per batch)
   - Separate methods for:
     a) Basic opinion loading (nodes only)
     b) Basic citation loading (relationships)
     c) Rich citation loading (relationships with metadata)

4. Schema Management:
   - Explicit schema definition and constraints
   - Centralized in Neo4jSchema class
   - Automatic constraint/index management
   - Range index on cluster_id for efficient lookups

5. Error Handling and Logging:
   - Comprehensive logging of operations and statistics
   - Graceful handling of missing or invalid data
   - Warning logs for skipped items
   - Statistics tracking for major operations

Usage Examples:
    # Initialize and configure
    loader = Neo4jLoader(uri, user, password)
    loader.initialize_database()  # Creates constraints and indexes

    # Load opinions (nodes) in batch
    opinions = [
        {
            'cluster_id': 1, 
            'date_filed': '2020-01-01',
            'case_name': 'Smith v. Jones',
            'docket_id': 123,
            'docket_number': '20-1234',
            'court_id': 5,
            'court_name': 'Supreme Court of the United States'
        }
    ]
    loader.load_opinions_batch(opinions)

    # Load basic citations (relationships)
    basic_citations = [(1, 2), (2, 3)]  # (citing_id, cited_id)
    loader.load_basic_citations(basic_citations, source="westlaw")

    # Load rich citations with metadata
    loader.load_citations_batch(rich_citations, source="ml_model_v1")

    # Database management
    loader.clear_database(source="westlaw")  # Clear specific source
    loader.clear_database()  # Clear everything
    loader.reset_database()  # Clear and reinitialize

Data Model:
1. Opinion Nodes (Label: Opinion)
   - Required properties:
     - cluster_id (INTEGER): Maps to 'id' in source database
     - date_filed (DATE): When the opinion was filed
     - case_name (STRING): Name of the case
     - docket_id (INTEGER): Foreign key to docket table
     - docket_number (STRING): Text representation of docket
     - court_id (INTEGER): Foreign key to court table
     - court_name (STRING): Text representation of court
   - Optional properties:
     - scdb_votes_majority (INTEGER): Supreme Court Database majority votes
     - scdb_votes_minority (INTEGER): Supreme Court Database minority votes
     - created_at (DATETIME): When node was created in Neo4j
     - updated_at (DATETIME): When node was last updated in Neo4j

2. Citation Relationships (Type: CITES)
   - Required: source (STRING), version (INTEGER)
   - Optional: treatment, relevance, reasoning, etc. (see Neo4jSchema.OPTIONAL_CITATION_PROPERTIES)
   - Metadata: timestamp

Note on Citation Sources:
- Each citation relationship includes a source property
- source: identifies where this citation data came from (e.g., "westlaw", "lexis", "ml_model_v1")
- This allows for:
  a) Tracking citation data provenance
  b) Easy filtering and analysis by source
  c) Ability to merge or compare citations from different sources
  d) Selective clearing/resetting of specific citation sets

Performance Considerations:
1. Batch Processing:
   - Default batch size of 50,000 items optimized for memory/performance balance
   - Separate node and relationship creation in basic citation loading
   - Progress logging for long-running operations

2. Database Optimization:
   - Unique constraint on cluster_id
   - Range index on cluster_id for efficient lookups
   - Indexes on commonly queried properties (date_filed, court_id)
   - Efficient versioning using max aggregation
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from src.llm_extraction.models import (
    CitationType,
    CombinedResolvedCitationAnalysis,
    CitationTreatment,
)
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jSchema:
    """Manages Neo4j schema definitions and constraints"""

    # Only cluster_id is truly required for the graph structure
    REQUIRED_OPINION_PROPERTIES = {
        "cluster_id": "INTEGER",  # Maps to 'id' in source database
    }

    # All other properties are optional
    OPTIONAL_OPINION_PROPERTIES = {
        "date_filed": "DATE",
        "case_name": "STRING",
        "docket_id": "INTEGER",
        "docket_number": "STRING",
        "court_id": "INTEGER",
        "court_name": "STRING",
        "opinion_type": "STRING",  # Type of opinion (e.g., majority, dissent)
        "scdb_votes_majority": "INTEGER",
        "scdb_votes_minority": "INTEGER",
        "created_at": "DATETIME",  # When this node was created in Neo4j
        "updated_at": "DATETIME",  # When this node was last updated in Neo4j
        "opinion_id": "INTEGER",  # Original database ID from CourtListener
        "docket_db_id": "INTEGER",  # Original docket ID from CourtListener
        "court_db_id": "INTEGER",  # Original court ID from CourtListener
    }

    # Required properties for Citation relationships
    REQUIRED_CITATION_PROPERTIES = {
        "source": "STRING"  # Source of this citation (e.g., westlaw, lexis, ml_model_v1)
    }

    # Optional citation metadata
    OPTIONAL_CITATION_PROPERTIES = {
        "treatment": "STRING",  # POSITIVE, NEGATIVE, etc.
        "relevance": "INTEGER",  # 1-4
        "reasoning": "STRING",
        "citation_text": "STRING",
        "page_number": "INTEGER",
        "timestamp": "DATETIME",  # When this citation was added
        "opinion_section": "STRING",  # majority, dissent, or concurrent
    }

    @staticmethod
    def get_constraint_queries():
        """Returns list of Cypher queries for creating all necessary constraints and indexes"""
        return [
            # Only enforce uniqueness on cluster_id
            """
            CREATE CONSTRAINT opinion_cluster_id IF NOT EXISTS 
            FOR (o:Opinion) REQUIRE o.cluster_id IS UNIQUE
            """,
            # Range index for cluster_id lookups (critical for MATCH performance)
            """
            CREATE RANGE INDEX opinion_cluster_id_range IF NOT EXISTS
            FOR (o:Opinion) ON (o.cluster_id)
            """,
        ]

    @staticmethod
    def filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from a dictionary"""
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def validate_opinion_properties(cls, data: Dict[str, Any]) -> None:
        """
        Validate opinion properties against the schema.
        Raises ValueError if any properties are invalid or unknown.
        """
        allowed_properties = set(cls.REQUIRED_OPINION_PROPERTIES.keys()) | set(
            cls.OPTIONAL_OPINION_PROPERTIES.keys()
        )
        unknown_properties = set(data.keys()) - allowed_properties
        if unknown_properties:
            raise ValueError(
                f"Unknown opinion properties not allowed in schema: {unknown_properties}"
            )

        # Validate required properties
        for key, expected_type in cls.REQUIRED_OPINION_PROPERTIES.items():
            if key not in data:
                raise ValueError(f"Required property missing: {key}")
            value = data[key]
            if expected_type == "INTEGER" and not isinstance(value, int):
                raise ValueError(f"Property {key} must be an integer")
            elif expected_type == "STRING" and not isinstance(value, str):
                raise ValueError(f"Property {key} must be a string")

        # Validate optional properties if present
        for key, value in data.items():
            if key in cls.OPTIONAL_OPINION_PROPERTIES and value is not None:
                expected_type = cls.OPTIONAL_OPINION_PROPERTIES[key]
                if expected_type == "INTEGER" and not isinstance(value, int):
                    raise ValueError(f"Property {key} must be an integer")
                elif expected_type == "STRING" and not isinstance(value, str):
                    raise ValueError(f"Property {key} must be a string")
                elif expected_type == "DATE":
                    if not isinstance(value, str):
                        raise ValueError(
                            f"Property {key} must be a date string in YYYY-MM-DD format"
                        )
                    try:
                        datetime.strptime(value, "%Y-%m-%d")
                    except ValueError:
                        raise ValueError(
                            f"Property {key} must be a date string in YYYY-MM-DD format"
                        )
                elif expected_type.startswith("LIST<") and not isinstance(value, list):
                    raise ValueError(f"Property {key} must be a list")


class Neo4jLoader:
    # Generate type_to_label mapping from CitationType enum
    # type_to_label = {
    #     CitationType.judicial_opinion.value: "Opinion",
    #     CitationType.statutes_codes_regulations.value: "Statute",
    #     CitationType.constitution.value: "Constitution",
    #     CitationType.electronic_resource.value: "ElectronicResource",
    #     CitationType.arbitration.value: "Arbitration",
    #     CitationType.court_rules.value: "CourtRules",
    #     CitationType.books.value: "Books",
    #     CitationType.law_journal.value: "LawJournal",
    #     CitationType.other.value: "OtherCitation",
    # }

    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.schema = Neo4jSchema()
        self.missing_citations = []

    def close(self):
        self.driver.close()

    def initialize_database(self):
        """Initialize database schema and constraints"""
        with self.driver.session() as session:
            # Create base indexes and constraints
            for query in self.schema.get_constraint_queries():
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Error creating constraint/index: {str(e)}")

            # Create constraints for each node type
            for node_type in CitationType:
                try:
                    # Ensure cluster_id is unique for each node type
                    session.run(
                        f"""
                        CREATE CONSTRAINT {node_type}_cluster_id IF NOT EXISTS 
                        FOR (n:{node_type}) REQUIRE n.cluster_id IS UNIQUE
                    """
                    )
                    # Create range index for cluster_id lookups
                    session.run(
                        f"""
                        CREATE RANGE INDEX {node_type}_cluster_id_range IF NOT EXISTS
                        FOR (n:{node_type}) ON (n.cluster_id)
                    """
                    )
                except Exception as e:
                    logger.warning(
                        f"Error creating constraint/index for {node_type}: {str(e)}"
                    )

            # Note: Relationship property constraints removed as they require Enterprise Edition
            logger.info("Initialized database schema, constraints, and indexes")

    def _process_basic_batch(self, session, batch: List[Tuple[int, int]], source: str):
        """Process a batch of basic citations using efficient map-based updates and optimized batching"""

        def _batch_transaction(tx):
            # Improved verification query that correctly counts total vs found
            verify_query = """
            WITH $batch as pairs
            UNWIND keys(pairs) as pair_key
            WITH pairs[pair_key] as pair, size(keys($batch)) as total_pairs
            OPTIONAL MATCH (citing:Opinion {cluster_id: pair[0]})
            OPTIONAL MATCH (cited:Opinion {cluster_id: pair[1]})
            WITH total_pairs, 
                 sum(CASE WHEN citing IS NOT NULL AND cited IS NOT NULL THEN 1 ELSE 0 END) as found_pairs
            RETURN total_pairs, found_pairs
            """

            batch_map = {str(i): pair for i, pair in enumerate(batch)}
            result = tx.run(verify_query, batch=batch_map)
            record = result.single()
            if record["found_pairs"] < record["total_pairs"]:
                raise ValueError(
                    f"Some opinions do not exist in the database ({record['found_pairs']}/{record['total_pairs']} found). "
                    "Please load all opinions first using load_opinions_batch."
                )

            # Then create the relationships with optimized batching
            rels_query = """
            WITH $batch as pairs
            UNWIND keys(pairs) as pair_key
            WITH pairs[pair_key] as pair
            MATCH (citing:Opinion {cluster_id: pair[0]})
            MATCH (cited:Opinion {cluster_id: pair[1]})
            
            // Use efficient aggregation for version counting
            WITH citing, cited, count((citing)-[:CITES]->(cited)) as citation_count
            WHERE citing IS NOT NULL AND cited IS NOT NULL
            
            MERGE (citing)-[r:CITES]->(cited)
            SET 
                r.source = $source,
                r.version = citation_count + 1,
                r.timestamp = datetime()
            """
            tx.run(rels_query, batch=batch_map, source=source)

        session.execute_write(_batch_transaction)

    def load_basic_citations(
        self,
        citation_pairs: List[Tuple[int, int]],
        source: str,
        batch_size: int = 50000,
    ):
        """
        Load basic citation relationships where we only know which opinion cites which.
        Assumes all opinion nodes have already been created using load_opinions_batch.

        Args:
            citation_pairs: List of (citing_cluster_id, cited_cluster_id) tuples
            source: Source of these citations (e.g., 'westlaw', 'lexis')
            batch_size: How many citations to process in each batch

        Raises:
            ValueError: If any referenced opinions don't exist in the database
        """
        with self.driver.session() as session:
            total_processed = 0
            batch = []

            for citing_id, cited_id in citation_pairs:
                batch.append((citing_id, cited_id))

                if len(batch) >= batch_size:
                    self._process_basic_batch(session, batch, source)
                    total_processed += len(batch)
                    logger.info(f"Processed {total_processed} citations")
                    batch = []

            # Process remaining items
            if batch:
                self._process_basic_batch(session, batch, source)
                total_processed += len(batch)
                logger.info(f"Final processed count: {total_processed} citations")

    def _create_opinion_node(self, tx, opinion_data: Dict[str, Any]):
        """Create a node with all available metadata"""
        # Filter out None values
        filtered_data = self.schema.filter_none_values(opinion_data)

        # Get the node type, defaulting to Opinion if not specified
        node_type = filtered_data.pop("type", CitationType.judicial_opinion.value)

        # Create node with appropriate label based on type
        query = f"""
        MERGE (o:`{node_type}` {{cluster_id: $cluster_id}})
        ON CREATE SET 
            o.created_at = datetime(),
            o.type = $type
        SET 
            o += $props,
            o.updated_at = datetime()
        RETURN o
        """

        try:
            result = tx.run(
                query,
                cluster_id=filtered_data["cluster_id"],
                type=node_type,
                props=filtered_data,
            )
            if not result.single():
                logger.error(
                    f"Failed to create/update node with cluster_id: {filtered_data['cluster_id']}"
                )
        except Exception as e:
            logger.error(f"Error creating node: {str(e)}")
            raise

    def _create_citation_relationship(
        self,
        tx,
        from_cluster_id: int,
        to_cluster_id: int,
        citation_data: Dict[str, Any],
    ):
        """
        Create a CITES relationship between nodes.
        If the cited node doesn't exist, create it with the appropriate type.
        Always creates the relationship with metadata.
        """
        filtered_data = self.schema.filter_none_values(citation_data)

        # Ensure required properties with defaults
        if "source" not in filtered_data:
            filtered_data["source"] = "unknown"
            logger.warning(
                f"Missing required 'source' property for citation: {citation_data}, using default 'unknown'"
            )

        # Initialize version and timestamp
        filtered_data["version"] = 1
        filtered_data["timestamp"] = datetime.now().isoformat()

        # Get citation type, defaulting to judicial_opinion
        citation_type = filtered_data.get("type", CitationType.judicial_opinion.value)

        # Create or match nodes and create relationship
        query = (
            """
        // Match or create the citing opinion (should always exist)
        MATCH (from:Opinion {cluster_id: $from_id})
        
        // Match or create the cited node with appropriate type
        MERGE (to:`%s` {cluster_id: $to_id})
        ON CREATE SET 
            to.created_at = datetime(),
            to.cluster_id = $to_id,
            to.type = $type
        
        // Create the relationship with all properties
        MERGE (from)-[r:CITES]->(to)
        SET r = $props
        RETURN from, to, r
        """
            % citation_type
        )

        try:
            result = tx.run(
                query,
                from_id=from_cluster_id,
                to_id=to_cluster_id,
                type=citation_type,
                props=filtered_data,
            )
            record = result.single()
            if not record:
                logger.error(
                    f"Failed to create citation relationship: from_id {from_cluster_id}, to_id {to_cluster_id}"
                )
        except Exception as e:
            logger.error(
                f"Error creating citation relationship: from_id {from_cluster_id}, to_id {to_cluster_id}, error: {str(e)}"
            )

    def save_missing_citations(self, filename: str = "missing_citations.json"):
        """Save missing citations to a JSON file for later analysis"""
        if not self.missing_citations:
            logger.info("No missing citations to save")
            return

        try:
            with open(filename, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "missing_citations": [
                            {"from_id": from_id, "to_id": to_id}
                            for from_id, to_id in self.missing_citations
                        ],
                    },
                    f,
                    indent=2,
                )
            logger.info(
                f"Saved {len(self.missing_citations)} missing citations to {filename}"
            )
        except Exception as e:
            logger.error(f"Failed to save missing citations: {str(e)}")

    def load_citations_batch(
        self,
        citations_data: List[CombinedResolvedCitationAnalysis],
        source: str,
        batch_size: int = 50000,
        preloaded_opinions: bool = False,
    ):
        """Load citations in batches with metadata"""
        with self.driver.session() as session:
            total_processed = 0
            batch = []

            for citation_analysis in citations_data:
                # Process citations from different sections
                sections = [
                    ("majority", citation_analysis.majority_opinion_citations),
                    ("concurrent", citation_analysis.concurrent_opinion_citations),
                    ("dissent", citation_analysis.dissenting_citations),
                ]

                for section_name, citations in sections:
                    for citation in citations:
                        if citation.resolved_opinion_cluster is None:
                            # Create a new node for non-opinion citations
                            if citation.type != CitationType.judicial_opinion.value:
                                citation_data = {
                                    "source": source,
                                    "opinion_section": section_name,
                                    "type": citation.type,
                                }

                                # Add optional fields if they exist
                                if citation.treatment:
                                    citation_data["treatment"] = citation.treatment
                                if citation.relevance:
                                    citation_data["relevance"] = citation.relevance
                                if citation.reasoning:
                                    citation_data["reasoning"] = citation.reasoning
                                if citation.citation_text:
                                    citation_data["citation_text"] = (
                                        citation.citation_text
                                    )
                                if citation.page_number:
                                    citation_data["page_number"] = citation.page_number

                                batch.append(
                                    {
                                        "from_id": citation_analysis.cluster_id,
                                        "to_id": hash(
                                            citation.citation_text
                                        ),  # Use hash as temporary ID
                                        "citation_data": citation_data,
                                    }
                                )
                            continue

                        # Regular opinion citation
                        citation_data = {
                            "source": source,
                            "opinion_section": section_name,
                            "type": citation.type,
                        }

                        # Add optional fields if they exist
                        if citation.treatment:
                            citation_data["treatment"] = citation.treatment
                        if citation.relevance:
                            citation_data["relevance"] = citation.relevance
                        if citation.reasoning:
                            citation_data["reasoning"] = citation.reasoning
                        if citation.citation_text:
                            citation_data["citation_text"] = citation.citation_text
                        if citation.page_number:
                            citation_data["page_number"] = citation.page_number

                        batch.append(
                            {
                                "from_id": citation_analysis.cluster_id,
                                "to_id": citation.resolved_opinion_cluster,
                                "citation_data": citation_data,
                            }
                        )

                        if len(batch) >= batch_size:
                            self._process_batch(session, batch)
                            total_processed += len(batch)
                            logger.info(f"Processed {total_processed} citations")
                            batch = []

            # Process remaining items
            if batch:
                self._process_batch(session, batch)
                total_processed += len(batch)
                logger.info(f"Final processed count: {total_processed} citations")

            if self.missing_citations:
                logger.info(
                    f"Total missing citations encountered: {len(self.missing_citations)}"
                )
                self.save_missing_citations()  # Save missing citations after processing

    def _process_batch(self, session, batch):
        """Process a batch of citations"""

        def _batch_transaction(tx):
            for item in batch:
                self._create_citation_relationship(
                    tx, item["from_id"], item["to_id"], item["citation_data"]
                )

        session.execute_write(_batch_transaction)

    def get_statistics(self) -> Dict[str, int]:
        """Get basic statistics about the database"""
        with self.driver.session() as session:
            result = session.run(
                """
                // First get node counts by type
                CALL {
                    MATCH (n) 
                    WITH labels(n) as nodeLabels
                    UNWIND nodeLabels as label
                    WITH label, count(*) as count
                    RETURN collect({label: label, count: count}) as node_counts
                }

                // Then get relationship stats
                CALL {
                    MATCH ()-[r:CITES]->()
                    RETURN count(r) as citation_count,
                           count(DISTINCT r.source) as source_count,
                           count(DISTINCT startNode(r).cluster_id) as citing_nodes,
                           count(DISTINCT endNode(r).cluster_id) as cited_nodes
                }

                // Return combined stats
                RETURN {
                    node_counts: node_counts,
                    citation_count: citation_count,
                    source_count: source_count,
                    citing_nodes: citing_nodes,
                    cited_nodes: cited_nodes
                } as stats
                """
            )
            stats = result.single()["stats"]
            logger.info(f"Database statistics: {stats}")
            return stats

    def load_opinions_batch(
        self,
        opinions: List[Dict[str, Any]],
        batch_size: int = 50000,
    ):
        """
        Load opinion nodes in batches.

        Args:
            opinions: List of dictionaries containing opinion data.
                     Each dict must have cluster_id field, all other fields are optional.
            batch_size: How many opinions to process in each batch.
                       Recommended to keep under 50k for optimal performance.
        """
        if batch_size > 50000:
            logger.warning(
                "Batch size > 50k may impact performance. Consider reducing batch size."
            )

        with self.driver.session() as session:
            total_processed = 0
            batch = []
            start_time = time.time()
            batch_times = []

            for opinion in opinions:
                # Only cluster_id is required
                if "cluster_id" not in opinion:
                    logger.warning(f"Skipping opinion without cluster_id: {opinion}")
                    continue

                # Filter out any None values
                filtered_opinion = {k: v for k, v in opinion.items() if v is not None}
                batch.append(filtered_opinion)

                if len(batch) >= batch_size:
                    batch_start = time.time()
                    self._process_opinions_batch(session, batch)
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)

                    total_processed += len(batch)
                    avg_batch_time = sum(batch_times) / len(batch_times)
                    logger.info(
                        f"Processed {total_processed} opinions. "
                        f"Last batch: {batch_time:.2f}s, "
                        f"Avg batch: {avg_batch_time:.2f}s"
                    )
                    batch = []

            # Process remaining items
            if batch:
                batch_start = time.time()
                self._process_opinions_batch(session, batch)
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                total_processed += len(batch)
                total_time = time.time() - start_time
                avg_batch_time = sum(batch_times) / len(batch_times)
                logger.info(
                    f"Final processed count: {total_processed} opinions. "
                    f"Total time: {total_time:.2f}s, "
                    f"Avg batch: {avg_batch_time:.2f}s"
                )

    def _process_opinions_batch(self, session, batch: List[Dict[str, Any]]):
        """Process a batch of opinions using efficient map-based updates"""

        def _batch_transaction(tx):
            for opinion in batch:
                self._create_opinion_node(tx, opinion)

        session.execute_write(_batch_transaction)


# Example usage:
if __name__ == "__main__":
    # Configuration
    NEO4J_URI = (
        "neo4j://localhost:7687/courtlistener"  # Updated to include database name
    )
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "courtlistener"  # Updated to match our password

    loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        # Initialize database (only needed once)
        loader.initialize_database()

        # Example loading opinions
        opinions = [
            {
                "cluster_id": 1001,
                "date_filed": "2020-01-01",
                "case_name": "Smith v. Jones",
                "docket_id": 123,
                "docket_number": "20-1234",
                "court_id": 5,
                "court_name": "Supreme Court of the United States",
                "scdb_votes_majority": 6,
                "scdb_votes_minority": 3,
            },
            {
                "cluster_id": 1002,
                "date_filed": "2020-02-01",
                "case_name": "Brown v. Board of Education",
                "docket_id": 124,
                "docket_number": "20-1235",
                "court_id": 5,
                "court_name": "Supreme Court of the United States",
            },
        ]
        loader.load_opinions_batch(opinions)

        # Example loading basic citation pairs
        basic_citations = [
            (1001, 1002),  # Opinion 1001 cites Opinion 1002
            (1001, 1003),  # Opinion 1001 also cites Opinion 1003
            (1002, 1004),  # Opinion 1002 cites Opinion 1004
        ]
        loader.load_basic_citations(basic_citations, source="westlaw")

        # Get statistics
        stats = loader.get_statistics()
        logger.info(f"Database statistics: {stats}")
    finally:
        loader.close()
