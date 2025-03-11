#!/usr/bin/env python3

import csv
import logging
from typing import Iterator

import psycopg2
import psycopg2.extras

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    "dbname": "courtlistener",
    "user": "courtlistener",
    "password": "postgrespassword",  # Add password matching the user
    "host": "localhost",
    "cursor_factory": psycopg2.extras.DictCursor,  # Returns results as dictionaries
}

# Output file configuration
OUTPUT_FILE = "neo4j_import/cl_opinion_cluster_nodes.csv"
BATCH_SIZE = 10000  # Number of records to process at once


def get_query() -> str:
    return """
        WITH first_opinions AS (
            SELECT DISTINCT ON (cluster_id) *
            FROM search_opinion
            ORDER BY cluster_id, 
                CASE 
                    WHEN type = '010combined' THEN 0
                    WHEN type = '020lead' THEN 1
                    ELSE 2
                END,
                id
        )
        SELECT  
            fo.cluster_id as cluster_id, 
            soc.date_filed as date_filed,
            soc.case_name as case_name,
            soc.scdb_votes_majority as scdb_votes_majority,
            soc.scdb_votes_minority as scdb_votes_minority,
            sd.case_name as docket_case_name,
            sd.docket_number as docket_number,
            sc.full_name as court_name,
            sd.id as docket_db_id,
            fo.id as opinion_db_id,
            fo.type as opinion_type,
            sd.court_id as court_db_id
        FROM first_opinions fo
        JOIN search_opinioncluster soc on fo.cluster_id = soc.id
        JOIN search_docket sd on soc.docket_id = sd.id
        JOIN search_court sc on sd.court_id = sc.id
        WHERE soc.precedential_status = 'Published'
    """


def fetch_records(cursor) -> Iterator[dict]:
    """Fetch records in batches using server-side cursor"""
    query = get_query()

    # Declare a named server-side cursor
    cursor.execute(f"DECLARE opinion_cursor CURSOR FOR {query}")

    while True:
        cursor.execute(f"FETCH {BATCH_SIZE} FROM opinion_cursor")
        records = cursor.fetchall()
        if not records:
            break
        for record in records:
            yield dict(record)


def write_csv(records: Iterator[dict], filename: str):
    """Write records to CSV file in Neo4j admin import format"""
    total_records = 0

    # Define fieldnames for Neo4j admin import format
    fieldnames = [
        ":LABEL",
        "cluster_id:ID",
        "soc_date_filed:DATE",
        "case_name:STRING",
        "soc_scdb_votes_majority:INT",
        "soc_scdb_votes_minority:INT",
        "docket_case_name:STRING",
        "docket_number:STRING",
        "court_name:STRING",
        "_search_docket_id:INT",
        "_search_opinion_id:INT",
        "_search_opinion_type:STRING",
        "_court_id:STRING",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for record in records:
            # Transform record to Neo4j admin format
            neo4j_record = {
                ":LABEL": "Opinion",
                "cluster_id:ID": record["cluster_id"],
                "soc_date_filed:DATE": record["date_filed"],
                "case_name:STRING": record["case_name"],
                "soc_scdb_votes_majority:INT": (
                    record["scdb_votes_majority"]
                    if record["scdb_votes_majority"]
                    else ""
                ),
                "soc_scdb_votes_minority:INT": (
                    record["scdb_votes_minority"]
                    if record["scdb_votes_minority"]
                    else ""
                ),
                "docket_case_name:STRING": record["docket_case_name"],
                "docket_number:STRING": record["docket_number"],
                "court_name:STRING": record["court_name"],
                "_search_docket_id:INT": record["docket_db_id"],
                "_search_opinion_id:INT": record["opinion_db_id"],
                "_search_opinion_type:STRING": record["opinion_type"],
                "_court_id:STRING": record["court_db_id"],
            }
            writer.writerow(neo4j_record)
            total_records += 1

            if total_records % BATCH_SIZE == 0:
                logger.info(f"Processed {total_records} records")

    logger.info(f"Completed processing {total_records} total records")


def main():
    logger.info("Starting opinion cluster export")
    conn = None
    cursor = None

    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Process and write records
        records = fetch_records(cursor)
        write_csv(records, OUTPUT_FILE)

        logger.info(f"Successfully exported data to {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"Error during export: {str(e)}")
        raise

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    main()
