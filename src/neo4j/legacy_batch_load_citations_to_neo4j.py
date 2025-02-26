#!/usr/bin/env python3

import json
import logging
from typing import List
from src.llm_extraction.models import CombinedResolvedCitationAnalysis
from neo4j.legacy_load_neo4j import Neo4jLoader
from src.postgres.db_utils import test_database_connection
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_citations_from_json(json_file: str, neo4j_loader: Neo4jLoader, source: str):
    """
    Load citations from a JSON file containing a list of CombinedResolvedCitationAnalysis objects.
    Creates nodes for non-opinion citations and relationships between all nodes.

    Args:
        json_file: Path to JSON file containing citation data
        neo4j_loader: Initialized Neo4jLoader instance
        source: Source identifier for the citations
    """
    if not source:
        raise ValueError("Source is required")

    logger.info(f"Loading citations from {json_file}")

    try:
        # First test database connection
        if not test_database_connection():
            logger.error(
                "Failed to connect to PostgreSQL database. Please check configuration."
            )
            sys.exit(1)

        with open(json_file, "r") as f:
            citations_data = json.load(f)

        # Convert JSON strings back to CombinedResolvedCitationAnalysis objects
        citation_analyses: list[CombinedResolvedCitationAnalysis] = []
        total_items = sum(len(citations_data[key]) for key in citations_data)

        with tqdm(total=total_items, desc="Parsing citations") as pbar:
            for key in citations_data.keys():
                for idx, item in enumerate(citations_data[key]):
                    try:
                        # Skip None items
                        if item is None:
                            logger.warning(f"Skipping None citation at {key} {idx}")
                            pbar.update(1)
                            continue

                        # Parse JSON if needed
                        if isinstance(item, str):
                            item = json.loads(item)

                        # Skip if no data
                        if not item:
                            logger.warning(f"Skipping empty citation at {key} {idx}")
                            pbar.update(1)
                            continue

                        # Create citation analysis
                        citation_analysis = (
                            CombinedResolvedCitationAnalysis.from_citations_json(
                                item, int(key)
                            )
                        )
                        if citation_analysis:
                            citation_analyses.append(citation_analysis)
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse citation at {key} {idx}: {str(e)}"
                        )
                    finally:
                        pbar.update(1)

        logger.info(f"Successfully parsed {len(citation_analyses)} citation analyses")

        # Process citations and create relationships
        logger.info("Loading citation relationships with metadata")
        neo4j_loader.load_citations_batch(citation_analyses, source=source)

        # Get final statistics
        stats = neo4j_loader.get_statistics()
        logger.info(f"Final database statistics: {stats}")

    except Exception as e:
        logger.error(f"Error loading citations: {str(e)}")
        neo4j_loader.save_missing_citations()
        raise


def main():
    # Neo4j configuration
    NEO4J_URI = "neo4j://localhost:7687/courtlistener"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "courtlistener"  # Change this to your actual password

    # Initialize Neo4j loader 
    loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # Initialize database (creates constraints and indexes)
        loader.initialize_database()

        # Load citations from JSON file
        json_file = (
            "responses_trial_20250222_231238.json"  # Change this to your input file
        )
        load_citations_from_json(json_file, loader, source="gemini_flash_2_0")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        loader.close()


if __name__ == "__main__":
    main()
