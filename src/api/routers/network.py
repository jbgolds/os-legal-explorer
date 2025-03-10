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
async def get_network(cluster_id: str, depth: int = 1, direction: str = "outgoing"):
    """
    Get a citation network centered on a specific document.

    Parameters:
    - cluster_id: The primary ID of the document to analyze
    - depth: How many layers of citations to include (default: 1)
    - direction: "outgoing" for documents cited by this document (default),
                "incoming" for documents that cite this document

    Returns:
    - Network graph with nodes and links
    """
    logger.info(
        f"Citation network requested for cluster_id: {cluster_id}, depth: {depth}, direction: {direction}"
    )

    if depth > 3:
        # Limit depth for performance reasons
        depth = 3
        logger.info(f"Depth limited to 3 for performance reasons")

    # Validate and normalize direction
    direction = direction.lower()
    if direction not in ["incoming", "outgoing"]:
        logger.warning(f"Invalid direction '{direction}', defaulting to 'outgoing'")
        direction = "outgoing"

    # Convert to Neo4j direction constant
    from neomodel import OUTGOING, INCOMING
    neo4j_direction = INCOMING if direction == "incoming" else OUTGOING

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
        if source_doc.primary_id != cluster_id:
            logger.warning(
                f"Source document ID {source_doc.primary_id} does not match cluster_id {cluster_id}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Source document ID {source_doc.primary_id} does not match cluster_id {cluster_id}",
            )

        # Process first level of citations using traversal
        related_docs = []

        # Define traversal parameters
        from neomodel import Traversal
        definition = dict(
            node_class=Opinion,  # Use Opinion as the target node class
            direction=neo4j_direction,
            relation_type="CITES",  # Use the CITES relationship type
            model=None  # No specific relationship model class
        )

        # Create and execute traversal
        citation_traversal = Traversal(
            source_doc, 
            Opinion.__label__,
            definition
        )
        
        direct_citations = citation_traversal.all()
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
                    rel = source_doc.cites.relationship(related_doc)
                    source_id, target_id = str(source_doc.primary_id), related_id
                else:
                    # Related document cites the source document
                    rel = related_doc.cites.relationship(source_doc)
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

            # Track for next level processing if depth > 1
            related_docs.append(related_doc)

        # Process additional levels if depth > 1
        current_depth = 1
        while current_depth < depth and related_docs:
            next_related_docs = []

            for current_doc in related_docs:
                # Skip if current_doc is None or doesn't have primary_id
                if not current_doc or not getattr(current_doc, "primary_id", None):
                    continue

                # Set up traversal for the next level
                next_level_traversal = Traversal(
                    current_doc,
                    Opinion.__label__,
                    definition  # Reuse the same definition
                )
                
                further_related_docs = next_level_traversal.all()

                for further_related_doc in further_related_docs:
                    # Skip if further_related_doc is None or doesn't have citation_string
                    if not further_related_doc or not getattr(further_related_doc, "citation_string", None):
                        continue

                    # Generate a node ID
                    further_related_id = (
                        getattr(further_related_doc, "primary_id", None)
                        or f"cite-{further_related_doc.citation_string}"
                    )
                    further_related_id = str(further_related_id)

                    # Determine source and target IDs based on direction
                    if direction == "outgoing":
                        source_id, target_id = str(current_doc.primary_id), further_related_id
                    else:
                        source_id, target_id = further_related_id, str(current_doc.primary_id)

                    # Skip if we've already processed this node
                    if further_related_id in nodes:
                        # But still check if we need to add a link
                        if not any(
                            l.source == source_id and l.target == target_id
                            for l in links
                        ):
                            links.append(
                                NetworkLink(
                                    source=source_id,
                                    target=target_id,
                                    type="CITES",
                                )
                            )
                        continue

                    # Add node
                    try:
                        network_node = NetworkNode.from_legal_doc(further_related_doc)
                        nodes[further_related_id] = network_node
                    except Exception as e:
                        logger.error(f"Error creating network node for {further_related_id}: {e}")
                        continue

                    # Add link
                    links.append(
                        NetworkLink(
                            source=source_id,
                            target=target_id,
                            type="CITES",
                        )
                    )

                    # Track for next level processing if not at max depth
                    if current_depth + 1 < depth:
                        next_related_docs.append(further_related_doc)

            # Move to next level
            related_docs = next_related_docs
            current_depth += 1

        # Construct the final graph
        graph = NetworkGraph(nodes=list(nodes.values()), links=links)
        logger.info(
            f"Returning network with {len(graph.nodes)} nodes and {len(graph.links)} links"
        )
        return graph

    except Exception as e:
        logger.error(f"Error building citation network: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error building citation network: {str(e)}"
        )
