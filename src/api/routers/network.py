from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from src.neo4j_db.models import Opinion, LegalDocument, CITATION_TYPE_TO_NODE_TYPE
from neomodel import db
from src.api.shared import templates

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
    label: str
    type: str  # Opinion, Statute, etc.
    year: Optional[int] = None
    court: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_legal_doc(cls, doc: LegalDocument) -> "NetworkNode":
        """Create a network node from a legal document."""
        # Extract year from date_filed if available
        year = None
        # Safely check for date_filed attribute and handle it appropriately
        if isinstance(doc, Opinion) and hasattr(doc, "date_filed") and doc.date_filed:
            try:
                # Convert to string and try to parse
                date_str = str(doc.date_filed)
                # Try different date formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
                    try:
                        date_obj = datetime.strptime(date_str, fmt).date()
                        year = date_obj.year
                        break
                    except ValueError:
                        continue

                # If we couldn't parse the date, try to extract year directly from string
                if year is None and len(date_str) >= 4:
                    # Try to find a 4-digit year in the string
                    import re

                    year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
                    if year_match:
                        year = int(year_match.group(0))
            except Exception as e:
                logger.debug(
                    f"Could not extract year from date: {doc.date_filed}, error: {e}"
                )

        # Create metadata dictionary with safely extracted properties
        metadata = {}

        # Add primary table
        if doc.primary_table:
            metadata["primary_table"] = str(doc.primary_table)

        # Add citation string
        if doc.citation_string:
            metadata["citation"] = str(doc.citation_string)

        # Handle Opinion-specific fields
        if isinstance(doc, Opinion):
            # Add case_name if available
            if doc.case_name:
                metadata["case_name"] = str(doc.case_name)

            # Add docket_number if available
            if doc.docket_number:
                metadata["docket_number"] = str(doc.docket_number)

        # Get court name if available (for Opinion)
        court = None
        if isinstance(doc, Opinion) and doc.court_name:
            court = str(doc.court_name)

        # Determine the best label to use
        label = None
        if isinstance(doc, Opinion) and doc.case_name:
            label = str(doc.case_name)
        elif hasattr(doc, "title") and doc.title:
            label = str(doc.title)
        else:
            label = str(doc.citation_string) or f"Document {doc.primary_id}"

        # Determine document type
        doc_type = str(doc.primary_table) if doc.primary_table else "Unknown"

        return cls(
            id=str(doc.primary_id) if doc.primary_id else f"doc-{id(doc)}",
            label=label,
            type=doc_type,
            year=year,
            court=court,
            metadata=metadata,
        )


class NetworkLink(BaseModel):
    """Link (edge) in the citation network graph visualization."""

    source: str
    target: str
    type: str = "CITES"
    treatment: Optional[str] = None
    relevance: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def create_from_relationship(
        source_id: str, rel_data: Dict, target_doc: LegalDocument
    ) -> "NetworkLink":
        """Create a network link from relationship data and target document."""
        metadata = {}

        # Extract properties from relationship data
        if rel_data.get("reasoning"):
            metadata["reasoning"] = rel_data["reasoning"]

        if rel_data.get("page_number") is not None:
            metadata["page_number"] = rel_data["page_number"]

        if rel_data.get("opinion_section"):
            metadata["opinion_section"] = rel_data["opinion_section"]

        if rel_data.get("citation_text"):
            metadata["citation_text"] = rel_data["citation_text"]

        if rel_data.get("data_source"):
            metadata["data_source"] = rel_data["data_source"]

        return NetworkLink(
            source=str(source_id),
            target=str(target_doc.primary_id),
            type="CITES",
            treatment=rel_data.get("treatment"),
            relevance=rel_data.get("relevance"),
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
    if depth > 3:
        # Limit depth for performance reasons
        depth = 3

    try:
        # Find the source document
        source_doc = Opinion.nodes.first_or_none(primary_id=cluster_id)

        if not source_doc:
            # Try to find in other document types if needed
            # For now, we'll just raise a 404
            raise HTTPException(
                status_code=404, detail=f"Document with ID {cluster_id} not found"
            )

        # Build the graph
        nodes = {}  # Dictionary to track unique nodes by ID
        links = []  # List to track all links

        # Start with the source document
        source_node = NetworkNode.from_legal_doc(source_doc)
        nodes[source_node.id] = source_node

        # Use Cypher to efficiently get outgoing citations with specified depth
        query = """
        MATCH path = (source:Opinion {primary_id: $cluster_id})-[r:CITES*1..{depth}]->(cited)
        WITH source, relationships(path) as rels, nodes(path) as ns
        RETURN source, rels, ns
        """

        results, _ = db.cypher_query(query, {"cluster_id": cluster_id, "depth": depth})

        # Process results
        for row in results:
            _, relationships, path_nodes = row

            # Skip first node (source) as we already have it
            for i, node in enumerate(path_nodes[1:], 1):
                # Convert Neo4j node to our domain model
                doc_label = list(node.labels)[0]  # Get the primary label

                # Determine the appropriate model class based on the label
                if doc_label in CITATION_TYPE_TO_NODE_TYPE:
                    node_class = CITATION_TYPE_TO_NODE_TYPE[doc_label]
                else:
                    # Default to LegalDocument if we can't find a specific class
                    node_class = LegalDocument

                # Inflate the node
                doc_node = node_class.inflate(node)

                # Add the node if not already present
                if str(doc_node.primary_id) not in nodes:
                    nodes[str(doc_node.primary_id)] = NetworkNode.from_legal_doc(
                        doc_node
                    )

                # The depth of this node in the citation chain
                citation_depth = i

            # Process relationships - each relationship in the path
            for i, rel in enumerate(relationships):
                # Source is either the original source or the previous node in the path
                if i == 0:
                    source_node_id = str(cluster_id)
                else:
                    # Get the source node from the path
                    source_node_id = str(
                        LegalDocument.inflate(path_nodes[i]).primary_id
                    )

                # Get the target node
                target_node = LegalDocument.inflate(path_nodes[i + 1])

                # Get the relationship properties
                rel_props = dict(rel)

                # Create a link
                link = NetworkLink.create_from_relationship(
                    source_id=source_node_id, rel_data=rel_props, target_doc=target_node
                )

                # Add depth information
                if link.metadata is None:
                    link.metadata = {}
                link.metadata["depth"] = i + 1

                links.append(link)

        # Create and return the graph
        return NetworkGraph(nodes=list(nodes.values()), links=links)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating network graph: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating network graph: {str(e)}"
        )


@router.get("/{cluster_id}/citation-network/filtered", response_model=NetworkGraph)
async def get_filtered_network(
    cluster_id: str,
    depth: int = 1,
    min_relevance: Optional[int] = None,
    treatment: Optional[str] = None,
    limit_nodes: int = 100,
):
    """
    Get a citation network with filtering options (outgoing citations only).

    Parameters:
    - cluster_id: The primary ID of the document to analyze
    - depth: How many layers of citations to include (default: 1)
    - min_relevance: Minimum relevance score for citations (1-4)
    - treatment: Filter by treatment type (POSITIVE, NEGATIVE, NEUTRAL, CAUTION)
    - limit_nodes: Maximum number of nodes to return (for performance)

    Returns:
    - Network graph with nodes and links
    """
    if depth > 3:
        # Limit depth for performance reasons
        depth = 3

    try:
        # Find the source document
        source_doc = Opinion.nodes.first_or_none(primary_id=cluster_id)

        if not source_doc:
            # Try to find in other document types if needed
            # For now, we'll just raise a 404
            raise HTTPException(
                status_code=404, detail=f"Document with ID {cluster_id} not found"
            )

        # Build the graph
        nodes = {}  # Dictionary to track unique nodes by ID
        links = []  # List to track all links

        # Start with the source document
        source_node = NetworkNode.from_legal_doc(source_doc)
        nodes[source_node.id] = source_node

        # Build the Cypher query with filters
        query = """
        MATCH path = (source:Opinion {primary_id: $cluster_id})-[r:CITES*1..{depth}]->(cited)
        WHERE 1=1
        """

        params = {"cluster_id": cluster_id, "depth": depth}

        # Add relevance filter if provided
        if min_relevance is not None:
            query += " AND ALL(rel IN r WHERE rel.relevance >= $min_relevance)"
            params["min_relevance"] = min_relevance

        # Add treatment filter if provided
        if treatment is not None:
            query += " AND ALL(rel IN r WHERE rel.treatment = $treatment)"
            params["treatment"] = treatment

        # Add limit
        query += " LIMIT $limit"
        params["limit"] = limit_nodes

        # Complete the query
        query += " WITH source, relationships(path) as rels, nodes(path) as ns RETURN source, rels, ns"

        # Execute the query
        results, _ = db.cypher_query(query, params)

        # Process results - same as in get_network
        for row in results:
            _, relationships, path_nodes = row

            # Add nodes from the path
            for i, node in enumerate(path_nodes[1:], 1):
                # Convert Neo4j node to our domain model
                doc_label = list(node.labels)[0]  # Get the primary label

                if doc_label in CITATION_TYPE_TO_NODE_TYPE:
                    node_class = CITATION_TYPE_TO_NODE_TYPE[doc_label]
                else:
                    node_class = LegalDocument

                doc_node = node_class.inflate(node)

                if str(doc_node.primary_id) not in nodes:
                    nodes[str(doc_node.primary_id)] = NetworkNode.from_legal_doc(
                        doc_node
                    )

                citation_depth = i

            # Process relationships
            for i, rel in enumerate(relationships):
                if i == 0:
                    source_node_id = str(cluster_id)
                else:
                    source_node_id = str(
                        LegalDocument.inflate(path_nodes[i]).primary_id
                    )

                target_node = LegalDocument.inflate(path_nodes[i + 1])
                rel_props = dict(rel)

                link = NetworkLink.create_from_relationship(
                    source_id=source_node_id, rel_data=rel_props, target_doc=target_node
                )

                if link.metadata is None:
                    link.metadata = {}
                link.metadata["depth"] = i + 1

                links.append(link)

        return NetworkGraph(nodes=list(nodes.values()), links=links)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating filtered network: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating filtered network: {str(e)}"
        )


@router.get("/{cluster_id}/citation-network/component", response_class=HTMLResponse)
async def get_network_component(request: Request, cluster_id: str):
    """
    Get the citation network visualization component as HTML.

    Parameters:
    - cluster_id: The primary ID of the document to visualize

    Returns:
    - HTML component for the citation network
    """
    try:
        # Check if the document exists and has citations
        source_doc = Opinion.nodes.first_or_none(primary_id=cluster_id)

        if not source_doc:
            # Return empty component if document not found
            return templates.TemplateResponse(
                "components/citation_network.html", {"request": request}
            )

        # Return the network visualization component
        return templates.TemplateResponse(
            "components/citation_network.html",
            {"request": request, "cluster_id": cluster_id},
        )

    except Exception as e:
        logger.error(f"Error rendering network component: {str(e)}")
        # Return error message in the component
        return templates.TemplateResponse(
            "components/citation_network.html", {"request": request, "error": str(e)}
        )
