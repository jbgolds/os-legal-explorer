#!/usr/bin/env python3
"""
Unpack Citation Relationships

This script unpacks metadata from existing CITES relationships in the Neo4j database
using neomodel. For each relationship that has other_metadata_versions, it creates 
new separate relationships with the appropriate metadata and then cleans up the 
original relationship.

Usage:
    python scripts/unpack_citation_relationships.py [--before YYYY-MM-DD] [--after YYYY-MM-DD] [--dry-run]

    
Sub queries example:

MATCH (a)-[r:CITES]->(b)
WHERE id(a) = $start_id AND id(b) = $end_id
RETURN collect(r) AS existing_relationships
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, NamedTuple, Optional, Tuple, Any, TypedDict, Union, List

from dotenv import load_dotenv
from neomodel import config, db
from tqdm import tqdm
from src.neo4j_db.neomodel_loader import NeomodelLoader

# Import models from the project
from src.neo4j_db.models import CitesRel, LegalDocument, Opinion

load_dotenv()

# Initialize neomodel connection
neomodel_loader = NeomodelLoader()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("unpack_relationships.log")
    ]
)
logger = logging.getLogger("unpack_citations")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unpack citation relationship metadata into separate relationships"
    )
    parser.add_argument(
        "--before",
        type=str,
        help="Process relationships created before this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--after",
        type=str,
        help="Process relationships created after this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of relationships to process in each batch"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


def build_date_filter(before_date: Optional[str], after_date: Optional[str]) -> Tuple[str, Dict]:
    """
    Build date filter conditions for Cypher query.
    
    Args:
        before_date: Optional date string in YYYY-MM-DD format
        after_date: Optional date string in YYYY-MM-DD format
        
    Returns:
        Tuple of (filter_clause, parameters)
    """
    conditions = []
    params = {}
    
    for date_str, operator, param_name in [
        (before_date, "<", "before_date"),
        (after_date, ">", "after_date")
    ]:
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                conditions.append(f"r.created_at {operator} ${param_name}")
                params[param_name] = dt.strftime("%Y-%m-%d")
            except ValueError:
                logger.error(f"Invalid date format for --{param_name.split('_')[0]}: {date_str}. Use YYYY-MM-DD.")
                raise ValueError(f"Invalid date format for --{param_name.split('_')[0]}: {date_str}. Use YYYY-MM-DD.")
    
    filter_clause = " AND " + " AND ".join(conditions) if conditions else ""
    return filter_clause, params

def prepare_relationship_properties(properties: Dict[str, Any], is_new: bool = True) -> Dict[str, Any]:
    """
    Prepare properties for a relationship, handling special fields.
    
    Args:
        properties: Original properties dictionary
        is_new: Whether this is for a new relationship (True) or updating existing (False)
        
    Returns:
        Cleaned properties dictionary
    """
    # Make a copy to avoid modifying the original
    props = properties.copy()
    
    # Remove metadata versions to avoid circular references
    if "other_metadata_versions" in props and is_new:
        props.pop("other_metadata_versions")

    # Format timestamps for JSON serialization
    for time_field in ["created_at", "updated_at"]:
        if time_field in props:
            if isinstance(props[time_field], (int, float)):
                dt = datetime.fromtimestamp(props[time_field])
                props[time_field] = dt.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(props[time_field], datetime):
                props[time_field] = props[time_field].strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(props[time_field], str):
                # Try to parse and standardize if it's already a string
                try:
                    dt = datetime.strptime(props[time_field], "%Y-%m-%d %H:%M:%S")
                    props[time_field] = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # If we can't parse it, leave it as is
                    pass
    
    # Set version for new relationships or increment for existing
    if is_new:
        props["version"] = 1
    else:
        props["version"] = props.get("version", 0) + 1
    
    # Set updated_at to current time
    props["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return props


class rel_queries_res(NamedTuple):
    r: CitesRel
    a: LegalDocument
    b: LegalDocument

def find_relationships_to_process(batch_size: int, date_filter: str, params: Dict) -> List[rel_queries_res]:
    """
    Find relationships with other_metadata_versions property.
    
    Args:
        batch_size: Number of relationships to return
        date_filter: Date filter clause for query
        params: Query parameters
        
    Returns:
        List of tuples containing (relationship, start_node, end_node)
    """
    query = (
        "MATCH (a)-[r:CITES]->(b) "
        "WHERE r.version > 1"
        f"{date_filter} "
        "RETURN r, a, b "
        f"LIMIT {batch_size}"
    )
    
    results, _ = db.cypher_query(query, params, resolve_objects=True)
    return results



def create_or_update_relationship(start_node: LegalDocument, end_node: LegalDocument, 
                                 properties: Dict[str, Any], citation_text: str, 
                                 page_number: Optional[Union[int, str]]) -> bool:
    """
    Create a new relationship or update an existing one with the same citation_text and page_number.
    
    Returns:
        True if successful, False otherwise
    """

    # Check if a relationship with the same citation_text and page_number already exists
    existing_rel = neomodel_loader.find_existing_relationship(citing_node=start_node, cited_node=end_node, citation_text=citation_text, page_number=page_number)
    
    if existing_rel:
        # Update existing relationship
        logger.debug(f"Updating existing relationship {existing_rel.element_id}")
        
        # Prepare properties for update
        existing_rel.update_history(properties)
        return True
    else:
        # Create new relationship
        logger.debug(f"Creating new relationship from {start_node.element_id} to {end_node.element_id} with citation_text {citation_text} and page_number {page_number}")
        
        # Prepare properties for the new relationship
        new_props = prepare_relationship_properties(properties, is_new=True)
        
        # TODO USE EXISTING FUNCTION TO CREATE..
        # # Create the relationship
        # create_query = (
        #     "MATCH (a), (b) "
        #     "WHERE id(a) = $start_id AND id(b) = $end_id "
        #     "CREATE (a)-[r:CITES]->(b) "
        #     "SET r = $properties "
        #     "RETURN id(r) as new_rel_id"
        # )
        # results, _ = db.cypher_query(create_query, {
        #     "start_id": start_node.element_id, 
        #     "end_id": end_node.element_id, 
        #     "properties": new_props
        # })
        # TODO WIP HERE!
       #  return results and results[0] and results[0][0] is not None
    
def update_original_relationship(rel: CitesRel, processed_indices: List[int], versions: List[Dict[str, Any]]) -> bool:
    """
    Update the original relationship after processing its metadata versions.
    
    Args:
        rel: The original relationship
        processed_indices: Indices of processed versions
        versions: All metadata versions
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get current version
        current_version = rel.version
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
       
        # Otherwise, selectively remove processed versions
        remaining_versions = [v for i, v in enumerate(versions) if i not in processed_indices]
        update_query = (
            "MATCH ()-[r]->() "
            "WHERE id(r) = $rel_id "
            "SET r.version = $new_version, "
            "    r.updated_at = $updated_at, "
            "    r.other_metadata_versions = $remaining_versions"
        )
        db.cypher_query(
            update_query, 
            {
                "rel_id": rel.element_id, 
                "new_version": current_version + 1,
                "updated_at": updated_at,
                "remaining_versions": remaining_versions
            }
        )
        
        return True
    except Exception as e:
        logger.error(f"Error updating original relationship {rel.element_id}: {str(e)}")
        return False

def validate_metadata_versions(rel: CitesRel) -> bool:
    """
    Validate that a relationship's other_metadata_versions property is a non-empty list of dictionaries.
    """
    try:
        versions = rel.other_metadata_versions
        
        # Check if versions exists and is a list
        if not versions or not isinstance(versions, list):
            logger.warning(f"Relationship {rel.element_id} has invalid other_metadata_versions: not a list or empty")
            return False
        
        # Check if all items in the list are dictionaries
        if not all(isinstance(v, dict) for v in versions):
            logger.warning(f"Relationship {rel.element_id} has non-dictionary items in other_metadata_versions")
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Error validating other_metadata_versions for relationship {rel.element_id}: {str(e)}")
        return False

def process_relationships(neomodel_loader: NeomodelLoader, date_filter: str, params: Dict, batch_size: int, dry_run: bool):
    """
    Process relationships with metadata versions.
    """
    # Get an estimate of total relationships to process for the progress bar
    
    
    if dry_run:
        logger.info("DRY RUN: No changes will be made")
    
    # Process in batches
    processed = 0
    created = 0
    errors = 0
    
    # Initialize progress bar with the estimated total
    with tqdm(desc="Processing relationships") as pbar:
        # Keep processing until we get an empty batch
        while True:
            # Fetch batch of relationships
            results = find_relationships_to_process(batch_size, date_filter, params)
            
            # Break the loop if no more relationships to process
            if not results:
                logger.info("No more relationships to process")
                break
            
            # Process each relationship in the batch
            batch_processed = 0
            batch_created = 0
            batch_errors = 0
            
            for record in results:
                rel = record.r  # relationship object
                start_node = record.a  # start node object
                end_node = record.b  # end node object
                
                try:
                    # Get the metadata versions directly
                    versions = rel.other_metadata_versions
                    
                    
                    # Skip if versions is not valid
                    if not versions or not isinstance(versions, list) or not all(isinstance(v, dict) for v in versions):
                        logger.warning(f"Skipping relationship {rel.element_id} - invalid other_metadata_versions with type {type(versions)}")
                        batch_processed += 1
                        continue
                    
                    citation_text = rel.citation_text
                    
                    logger.debug(f"Processing relationship {rel.element_id} with {len(versions)} metadata versions")
                    
                    if not dry_run:
                        # Use a transaction for each relationship to ensure atomicity
                        with db.transaction:
                            # Track which versions were successfully processed
                            processed_indices = []
                            
                            # Create new relationships for each metadata version
                            for i, version_props in enumerate(versions):
                                # Add citation_text if it exists in the original relationship
                                if citation_text and "citation_text" not in version_props:
                                    version_props["citation_text"] = citation_text
                                
                                # Get the citation_text and page_number for this version
                                version_citation_text = version_props.get("citation_text", citation_text)
                                version_page_number = version_props.get("page_number")
                                
                                # Create or update relationship
                                if create_or_update_relationship(
                                    start_node, end_node, version_props, 
                                    version_citation_text, version_page_number
                                ):
                                    batch_created += 1
                                    processed_indices.append(i)
                            
                            # Update the original relationship
                            if processed_indices:
                                update_original_relationship(rel, processed_indices, versions)
                    
                    batch_processed += 1
                except Exception as e:
                    logger.error(f"Error processing relationship {rel.element_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    batch_errors += 1
            
            # Update counts
            processed += batch_processed
            created += batch_created
            errors += batch_errors
            
            # Update progress bar
            pbar.update(batch_processed)
            
            # Update progress bar description with current stats
            pbar.set_description(f"Processed: {processed}, Created: {created}, Errors: {errors}")
    
    # Summary
    logger.info(f"Processing complete:")
    logger.info(f"  - Processed: {processed} relationships")
    logger.info(f"  - Created: {created} new relationships")
    logger.info(f"  - Errors: {errors}")
    
    if dry_run:
        logger.info("DRY RUN: No changes were made")

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    

    
    
    # Get date filter conditions
    date_filter, params = build_date_filter(args.before, args.after)
    
    # Process relationships
    try:
        process_relationships(
            neomodel_loader, 
            date_filter, 
            params, 
            args.batch_size, 
            args.dry_run
        )
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 