import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.neo4j_db.models import LegalDocument, Opinion
import re
from neomodel import Traversal, INCOMING, OUTGOING, AsyncTraversal

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
async def get_network(cluster_id: str, direction: str = "outgoing"):
    """
    Get a citation network centered on a specific document.

    Parameters:
    - cluster_id: The primary ID of the document to analyze
    - direction: "outgoing" for documents cited by this document (default),
                "incoming" for documents that cite this document

    Returns:
    - Network graph with nodes and links
    """
    logger.info(
        f"Citation network requested for cluster_id: {cluster_id}, direction: {direction}"
    )

    # Validate and normalize direction
    direction = direction.lower()
    if direction not in ["incoming", "outgoing"]:
        logger.warning(f"Invalid direction '{direction}', defaulting to 'outgoing'")
        direction = "outgoing"

    # Convert to Neo4j direction constant
    
    neo4j_direction = INCOMING if direction == "incoming" else OUTGOING

    try:
        # Find the source document - await the coroutine
        source_doc = await Opinion.nodes.first_or_none(primary_id=cluster_id)

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
        if source_doc.primary_id != cluster_id:
            logger.warning(
                f"Source document ID {source_doc.primary_id} does not match cluster_id {cluster_id}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Source document ID {source_doc.primary_id} does not match cluster_id {cluster_id}",
            )

        # Define traversal parameters
        definition = dict(
            node_class=LegalDocument,  # Use LegalDocument as the target node class
            direction=neo4j_direction,
            relation_type="CITES",  # Use the CITES relationship type
            model=None  # No specific relationship model class
        )

        # Create and execute traversal
        citation_traversal = AsyncTraversal(
            source_doc, 
            LegalDocument.__label__,
            definition
        )
        
        # Get direct citations - now using await with AsyncTraversal
        direct_citations = await citation_traversal.all()
        logger.info(f"Found {len(direct_citations)} {direction} citations")

        # Process each citation based on direction
        for related_doc in direct_citations:
            # Skip if related_doc is None or doesn't have citation_string
            if not related_doc or not getattr(related_doc, "citation_string", None):
                logger.warning(f"Skipping invalid related document: {related_doc}")
                continue

            # Generate a node ID - use primary_id if available, otherwise use citation_string
            related_id = (
                getattr(related_doc, "primary_id", None)
                or f"cite-{related_doc.citation_string}"
            )
            related_id = str(related_id)

            # Add related node if not already added
            if related_id not in nodes:
                try:
                    network_node = NetworkNode.from_legal_doc(related_doc)
                    nodes[related_id] = network_node
                except Exception as e:
                    logger.error(f"Error creating network node for {related_id}: {e}")
                    continue

            # Get relationship properties and create link
            try:
                # The relationship direction affects how we get the relationship and create links
                if direction == "outgoing":
                    # Source document cites the related document
                    rel = await source_doc.cites.relationship(related_doc)
                    source_id, target_id = str(source_doc.primary_id), related_id
                else:
                    # Related document cites the source document
                    rel = await related_doc.cites.relationship(source_doc)
                    source_id, target_id = related_id, str(source_doc.primary_id)

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
                    source=source_id,
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
                if direction == "outgoing":
                    link = NetworkLink(
                        source=str(source_doc.primary_id), target=related_id, type="CITES"
                    )
                else:
                    link = NetworkLink(
                        source=related_id, target=str(source_doc.primary_id), type="CITES"
                    )

            links.append(link)

        # Construct the final graph
        graph = NetworkGraph(nodes=list(nodes.values()), links=links)
        logger.info(
            f"Returning network with {len(graph.nodes)} nodes and {len(graph.links)} links"
        )
        return graph

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error building citation network: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error building citation network: {str(e)}"
        )
