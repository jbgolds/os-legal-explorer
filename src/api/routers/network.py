from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from src.neo4j_db.models import Opinion, LegalDocument, CITATION_TYPE_TO_NODE_TYPE
from src.llm_extraction.models import CitationType
from neomodel import db

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/opinion",
    tags=["network"],
    responses={404: {"description": "Not found"}},
)


# Models for network visualization
class NetworkNode(BaseModel):
    """Node in the citation network graph visualization."""

    id: str
    type: str  # Opinion, Statute, etc.
    year: Optional[int] = None
    court: Optional[str] = None
    court_name: Optional[str] = None
    case_name: Optional[str] = None
    citation_string: Optional[str] = None
    docket_number: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_legal_doc(cls, doc: LegalDocument | Opinion) -> "NetworkNode":
        """Create a network node from a legal document."""
        metadata = {}

        # Extract basic fields with simple getattr calls
        year = cls._extract_year_from_date(getattr(doc, "date_filed", None))
        court = getattr(doc, "court_id", None)
        court_name = getattr(doc, "court_name", None)
        case_name = getattr(doc, "case_name", None)
        docket_number = getattr(doc, "docket_number", None)
        citation_string = getattr(doc, "citation_string", None)

        # Add basic metadata
        for field in ["docket_id", "opinion_id", "date_filed", "primary_table"]:
            value = getattr(doc, field, None)
            if value:
                metadata[field] = str(value)

        # Add additional metadata from doc.metadata if it exists
        if hasattr(doc, "metadata") and doc.metadata:
            try:
                doc_metadata = doc.metadata
                if isinstance(doc_metadata, dict):
                    for key, value in doc_metadata.items():
                        if key not in metadata and value is not None:
                            metadata[key] = str(value)
            except Exception as e:
                logger.warning(f"Error extracting document metadata: {e}")

        # Generate a node ID - use primary_id if available, otherwise use citation_string
        doc_type = getattr(doc, "primary_table", "Unknown")
        node_id = getattr(doc, "primary_id", None) or f"cite-{citation_string}"

        return cls(
            id=str(node_id),
            type=str(doc_type),
            year=year,
            court=court,
            court_name=court_name,
            case_name=case_name,
            docket_number=docket_number,
            citation_string=citation_string,
            metadata=metadata,
        )

    @staticmethod
    def _extract_year_from_date(date_value) -> Optional[int]:
        """Extract year from a date value that could be in various formats."""
        try:
            # Convert to string and try to parse
            date_str = str(date_value)

            # Try different date formats
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
                try:
                    date_obj = datetime.strptime(date_str, fmt).date()
                    return date_obj.year
                except ValueError:
                    continue

            # If we couldn't parse the date, try to extract year directly from string
            if len(date_str) >= 4:
                # Try to find a 4-digit year in the string
                import re

                year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
                if year_match:
                    return int(year_match.group(0))
        except Exception as e:
            logger.debug(f"Could not extract year from date: {date_value}, error: {e}")

        return None

    @staticmethod
    def _add_to_metadata_if_exists(
        metadata: Dict,
        doc: LegalDocument,
        attr_name: str,
        metadata_key: Optional[str] = None,
    ):
        """Add an attribute to metadata if it exists on the document."""
        if hasattr(doc, attr_name) and getattr(doc, attr_name) is not None:
            key = metadata_key or attr_name
            metadata[key] = str(getattr(doc, attr_name))


class NetworkLink(BaseModel):
    """Link (edge) in the citation network graph visualization."""

    source: str
    target: str
    type: str = "CITES"
    treatment: Optional[str] = None
    reasoning: Optional[str] = None
    relevance: Optional[int] = None
    section: Optional[str] = None  # MAJORITY, DISSENTING, CONCURRING
    metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def create_from_relationship(
        source_id: str, rel_data: Dict, target_doc: LegalDocument
    ) -> "NetworkLink":
        """Create a network link from relationship data and target document."""
        metadata = {}

        # Extract properties from relationship data
        for field in ["page_number", "citation_text", "data_source"]:
            if rel_data.get(field) is not None:
                metadata[field] = rel_data[field]

        # Generate a target ID - use primary_id if available, otherwise use citation_string
        target_id = (
            getattr(target_doc, "primary_id", None)
            or f"cite-{target_doc.citation_string}"
        )

        return NetworkLink(
            source=str(source_id),
            target=str(target_id),
            type="CITES",
            treatment=rel_data.get("treatment"),
            relevance=rel_data.get("relevance"),
            reasoning=rel_data.get("reasoning"),
            section=rel_data.get("opinion_section"),
            metadata=metadata,
        )


class NetworkGraph(BaseModel):
    """Complete graph for visualization with nodes and links."""

    nodes: List[NetworkNode]
    links: List[NetworkLink]


@router.get("/{cluster_id}/citation-network", response_model=NetworkGraph)
async def get_network(cluster_id: str, depth: int = 1):
    """
    Get a citation network centered on a specific document (outgoing citations only).

    Parameters:
    - cluster_id: The primary ID of the document to analyze
    - depth: How many layers of citations to include (default: 1)

    Returns:
    - Network graph with nodes and links
    """
    logger.info(
        f"Citation network requested for cluster_id: {cluster_id}, depth: {depth}"
    )

    if depth > 3:
        # Limit depth for performance reasons
        depth = 3
        logger.info(f"Depth limited to 3 for performance reasons")

    try:
        # Find the source document
        source_doc = Opinion.nodes.first_or_none(primary_id=cluster_id)

        if not source_doc:
            # Try to find in other document types if needed
            # For now, we'll just raise a 404
            logger.warning(f"Document with ID {cluster_id} not found")
            raise HTTPException(
                status_code=404, detail=f"Document with ID {cluster_id} not found"
            )

        # Build the graph
        nodes = {}  # Dictionary to track unique nodes by ID
        links = []  # List to track all links

        # Start with the source document
        source_node = NetworkNode.from_legal_doc(source_doc)
        nodes[source_node.id] = source_node

        logger.info(
            f"Source document found: {source_doc.primary_id}, should match {cluster_id}"
        )

        # Process first level of citations directly using Neomodel
        cited_docs = []

        # Get direct citations from the source document
        direct_citations = source_doc.cites.all()
        logger.info(f"Found {len(direct_citations)} direct citations")

        # Process each citation
        for target_doc in direct_citations:
            # Skip if target_doc is None or doesn't have citation_string
            if not target_doc or not getattr(target_doc, "citation_string", None):
                logger.warning(f"Skipping invalid target document: {target_doc}")
                continue

            # Generate a node ID - use primary_id if available, otherwise use citation_string
            target_id = (
                getattr(target_doc, "primary_id", None)
                or f"cite-{target_doc.citation_string}"
            )
            target_id = str(target_id)

            # Add target node if not already added
            if target_id not in nodes:
                try:
                    network_node = NetworkNode.from_legal_doc(target_doc)
                    nodes[target_id] = network_node
                except Exception as e:
                    logger.error(f"Error creating network node for {target_id}: {e}")
                    continue

            # Get relationship properties
            try:
                # Get the relationship object between source_doc and target_doc
                rel = source_doc.cites.relationship(target_doc)

                # Extract relationship properties
                rel_data = {}
                if rel:
                    for prop in [
                        "treatment",
                        "relevance",
                        "reasoning",
                        "citation_text",
                        "page_number",
                        "opinion_section",
                        "data_source",
                    ]:
                        value = getattr(rel, prop, None)
                        if value is not None:
                            rel_data[prop] = value

                # Create metadata dictionary
                metadata = {}
                for field in [
                    "citation_text",
                    "page_number",
                    "opinion_section",
                    "data_source",
                ]:
                    if field in rel_data:
                        metadata[field] = rel_data[field]

                # Create the link with relationship data
                link = NetworkLink(
                    source=str(source_doc.primary_id),
                    target=target_id,
                    type="CITES",
                    treatment=rel_data.get("treatment"),
                    reasoning=rel_data.get("reasoning"),
                    relevance=rel_data.get("relevance"),
                    section=rel_data.get("section"),
                    metadata=metadata if metadata else None,
                )
            except Exception as e:
                logger.warning(
                    f"Error extracting relationship data: {e}, creating basic link"
                )
                # Create a basic link if relationship data extraction fails
                link = NetworkLink(
                    source=str(source_doc.primary_id), target=target_id, type="CITES"
                )

            links.append(link)

            # Track for next level processing if depth > 1
            cited_docs.append(target_doc)

        # Process additional levels if depth > 1
        current_depth = 1
        while current_depth < depth and cited_docs:
            next_cited_docs = []

            for current_doc in cited_docs:
                # Skip if current_doc is None or doesn't have citation_string
                if not current_doc or not getattr(current_doc, "citation_string", None):
                    continue

                # Generate a node ID for the current document
                current_id = (
                    getattr(current_doc, "primary_id", None)
                    or f"cite-{current_doc.citation_string}"
                )
                current_id = str(current_id)

                # Get citations from this document
                try:
                    next_level_citations = current_doc.cites.all()
                except Exception as e:
                    logger.error(f"Error getting citations for {current_id}: {e}")
                    continue

                for target_doc in next_level_citations:
                    # Skip if target_doc is None or doesn't have citation_string
                    if not target_doc or not getattr(
                        target_doc, "citation_string", None
                    ):
                        continue

                    # Generate a node ID for the target document
                    target_id = (
                        getattr(target_doc, "primary_id", None)
                        or f"cite-{target_doc.citation_string}"
                    )
                    target_id = str(target_id)

                    # Add target node if not already added
                    if target_id not in nodes:
                        try:
                            network_node = NetworkNode.from_legal_doc(target_doc)
                            nodes[target_id] = network_node
                        except Exception as e:
                            logger.error(
                                f"Error creating network node for {target_id}: {e}"
                            )
                            continue

                    # Get relationship properties
                    try:
                        # Get the relationship object between current_doc and target_doc
                        rel = current_doc.cites.relationship(target_doc)

                        # Extract relationship properties
                        rel_data = {}
                        if rel:
                            for prop in [
                                "treatment",
                                "relevance",
                                "reasoning",
                                "citation_text",
                                "page_number",
                                "opinion_section",
                                "data_source",
                            ]:
                                value = getattr(rel, prop, None)
                                if value is not None:
                                    rel_data[prop] = value

                        # Create metadata dictionary
                        metadata = {}
                        for field in [
                            "citation_text",
                            "page_number",
                            "data_source",
                        ]:
                            if field in rel_data:
                                metadata[field] = rel_data[field]

                        # Create the link with relationship data
                        link = NetworkLink(
                            source=current_id,
                            target=target_id,
                            type="CITES",
                            treatment=rel_data.get("treatment"),
                            reasoning=rel_data.get("reasoning"),
                            relevance=rel_data.get("relevance"),
                            section=rel_data.get("section"),
                            metadata=metadata if metadata else None,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error extracting relationship data: {e}, creating basic link"
                        )
                        # Create a basic link if relationship data extraction fails
                        link = NetworkLink(
                            source=current_id, target=target_id, type="CITES"
                        )

                    links.append(link)

                    # Add to next level processing
                    next_cited_docs.append(target_doc)

            # Update for next iteration
            cited_docs = next_cited_docs
            current_depth += 1

        # Return the network graph
        network = NetworkGraph(nodes=list(nodes.values()), links=links)
        logger.info(
            f"Returning network with {len(network.nodes)} nodes and {len(network.links)} links"
        )
        return network

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_network: {e}", exc_info=True)
        # Return a minimal network with just the source node
        if "source_node" in locals() and source_node:
            return NetworkGraph(nodes=[source_node], links=[])
        else:
            # If we couldn't even create the source node, return an empty network
            return NetworkGraph(nodes=[], links=[])
