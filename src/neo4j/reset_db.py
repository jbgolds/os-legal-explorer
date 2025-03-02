#!/usr/bin/env python3
"""
Reset Neo4j Database

This script provides a complete reset of the Neo4j database,
removing all nodes, relationships, and schema elements.
"""

import os
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "courtlistener")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def reset_database():
    """
    Reset the entire Neo4j database by:
    1. Removing all nodes and relationships
    2. Dropping all constraints and indexes
    """
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # Step 1: Delete all nodes and relationships
            logger.info("Deleting all nodes and relationships...")
            session.run("MATCH (n) DETACH DELETE n")

            # Step 2: Drop all constraints and indexes using APOC
            logger.info("Dropping all constraints and indexes...")
            try:
                session.run("CALL apoc.schema.assert({},{},true)")
                logger.info("Successfully reset schema using APOC")
            except Exception as e:
                logger.warning(
                    f"APOC schema reset failed (you may need to drop constraints manually): {str(e)}"
                )

                # Fallback to manual dropping of constraints and indexes
                # Get all constraints
                constraints = list(session.run("SHOW CONSTRAINTS"))
                for constraint in constraints:
                    try:
                        constraint_name = constraint.get("name")
                        if constraint_name:
                            session.run(f"DROP CONSTRAINT {constraint_name}")
                            logger.info(f"Dropped constraint: {constraint_name}")
                    except Exception as e:
                        logger.error(f"Failed to drop constraint: {str(e)}")

                # Get all indexes
                indexes = list(session.run("SHOW INDEXES"))
                for index in indexes:
                    try:
                        index_name = index.get("name")
                        if index_name:
                            session.run(f"DROP INDEX {index_name}")
                            logger.info(f"Dropped index: {index_name}")
                    except Exception as e:
                        logger.error(f"Failed to drop index: {str(e)}")

            # Print database statistics after reset
            count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            logger.info(f"Database reset complete. Node count: {count}")

            # List available labels (if any remain)
            labels = list(session.run("CALL db.labels()"))
            if labels:
                label_list = [label[0] for label in labels]
                logger.warning(f"Labels still present in database: {label_list}")
                logger.info(
                    "NOTE: In Neo4j, label definitions persist in metadata even when no nodes have those labels."
                )
                logger.info(
                    "This is normal and doesn't affect functionality - new data can reuse these labels."
                )

                # Verify that no nodes actually have these labels
                for label in label_list:
                    count = session.run(
                        f"MATCH (n:`{label}`) RETURN count(n) as count"
                    ).single()["count"]
                    if count > 0:
                        logger.error(
                            f"Found {count} nodes with label '{label}' that weren't deleted!"
                        )
                    else:
                        logger.debug(f"Confirmed no nodes with label '{label}' exist")

                logger.info(
                    "To completely remove label definitions, a database recreation is required:"
                )
                logger.info("1. Stop Neo4j server")
                logger.info(
                    "2. Delete the database files (typically in data/databases/)"
                )
                logger.info("3. Restart Neo4j server")
            else:
                logger.info("No labels remain in the database")

    finally:
        driver.close()


if __name__ == "__main__":
    reset_database()
