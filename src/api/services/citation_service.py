from typing import List, Dict, Optional, Any
import logging

from ..models.citations import CitationNetwork, CitationDetail, CitationStats, Node, Edge
from src.neo4j.models import Opinion as Neo4jOpinion, CitesRel

logger = logging.getLogger(__name__)

def get_citation_network(
    neo4j_session,
    cluster_id: Optional[int] = None,
    court_id: Optional[str] = None,
    depth: int = 1,
    limit: int = 100
) -> CitationNetwork:
    """
    Get a citation network centered around a specific opinion or court.
    
    Args:
        neo4j_session: Neo4j session
        cluster_id: Center the network on this opinion cluster ID
        court_id: Filter by court ID
        depth: Network depth (1-3)
        limit: Maximum number of nodes to return
        
    Returns:
        Citation network with nodes and edges
    """
    # Validate depth
    if depth < 1 or depth > 3:
        depth = 1  # Default to 1 if invalid
    
    try:
        nodes = []
        edges = []
        
        if cluster_id:
            # Get the center opinion
            try:
                center_opinion = Neo4jOpinion.nodes.get(cluster_id=cluster_id)
                
                # Add center node
                center_node = Node.from_neo4j(center_opinion, node_type="center")
                nodes.append(center_node)
                
                # Get outgoing citations (opinions cited by this opinion)
                cited_opinions = []
                for i in range(depth):
                    if i == 0:
                        # First level: direct citations
                        new_cited = center_opinion.cites.all()[:limit]
                        cited_opinions.extend(new_cited)
                    else:
                        # Deeper levels: citations of citations
                        next_level = []
                        for opinion in cited_opinions:
                            next_level.extend(opinion.cites.all()[:limit // (i + 1)])
                        cited_opinions.extend(next_level)
                
                # Add cited nodes and edges
                for cited in cited_opinions:
                    # Add node if not already added
                    if not any(n.id == cited.cluster_id for n in nodes):
                        nodes.append(Node.from_neo4j(cited))
                    
                    # Find the relationship and add edge
                    for citing in [o for o in cited_opinions + [center_opinion] if hasattr(o, 'cites')]:
                        rel = citing.cites.relationship(cited)
                        if rel:
                            edges.append(Edge.from_neo4j_rel(rel, citing.cluster_id, cited.cluster_id))
                
                # Get incoming citations (opinions citing this opinion)
                citing_opinions = []
                for i in range(depth):
                    if i == 0:
                        # First level: direct citations
                        new_citing = center_opinion.cited_by.all()[:limit]
                        citing_opinions.extend(new_citing)
                    else:
                        # Deeper levels: citations of citations
                        next_level = []
                        for opinion in citing_opinions:
                            next_level.extend(opinion.cited_by.all()[:limit // (i + 1)])
                        citing_opinions.extend(next_level)
                
                # Add citing nodes and edges
                for citing in citing_opinions:
                    # Add node if not already added
                    if not any(n.id == citing.cluster_id for n in nodes):
                        nodes.append(Node.from_neo4j(citing))
                    
                    # Find the relationship and add edge
                    rel = citing.cites.relationship(center_opinion)
                    if rel:
                        edges.append(Edge.from_neo4j_rel(rel, citing.cluster_id, center_opinion.cluster_id))
                    
                    # Add edges between citing opinions
                    for cited in [o for o in citing_opinions + [center_opinion] if hasattr(citing, 'cites')]:
                        rel = citing.cites.relationship(cited)
                        if rel:
                            edges.append(Edge.from_neo4j_rel(rel, citing.cluster_id, cited.cluster_id))
            
            except Neo4jOpinion.DoesNotExist:
                logger.warning(f"Opinion with cluster_id {cluster_id} not found")
        
        elif court_id:
            # Get opinions from this court
            court_opinions = Neo4jOpinion.nodes.filter(court_id=court_id)[:limit]
            
            # Add court opinion nodes
            for opinion in court_opinions:
                nodes.append(Node.from_neo4j(opinion, node_type="court"))
            
            # Get outgoing citations from court opinions
            for opinion in court_opinions:
                cited_opinions = opinion.cites.all()[:limit // len(court_opinions)]
                
                # Add cited nodes and edges
                for cited in cited_opinions:
                    # Add node if not already added
                    if not any(n.id == cited.cluster_id for n in nodes):
                        nodes.append(Node.from_neo4j(cited))
                    
                    # Find the relationship and add edge
                    rel = opinion.cites.relationship(cited)
                    if rel:
                        edges.append(Edge.from_neo4j_rel(rel, opinion.cluster_id, cited.cluster_id))
            
            # Get incoming citations to court opinions
            for opinion in court_opinions:
                citing_opinions = opinion.cited_by.all()[:limit // len(court_opinions)]
                
                # Add citing nodes and edges
                for citing in citing_opinions:
                    # Add node if not already added
                    if not any(n.id == citing.cluster_id for n in nodes):
                        nodes.append(Node.from_neo4j(citing))
                    
                    # Find the relationship and add edge
                    rel = citing.cites.relationship(opinion)
                    if rel:
                        edges.append(Edge.from_neo4j_rel(rel, citing.cluster_id, opinion.cluster_id))
        
        # Create network
        network = CitationNetwork(
            nodes=nodes,
            edges=edges,
            metadata={
                "center_id": cluster_id,
                "court_id": court_id,
                "depth": depth,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        )
        
        return network
    
    except Exception as e:
        logger.error(f"Error getting citation network: {str(e)}")
        # Return empty network on error
        return CitationNetwork(nodes=[], edges=[])

def get_citation_detail(
    neo4j_session,
    citing_id: int,
    cited_id: int
) -> Optional[CitationDetail]:
    """
    Get detailed information about a specific citation relationship.
    
    Args:
        neo4j_session: Neo4j session
        citing_id: The citing opinion cluster ID
        cited_id: The cited opinion cluster ID
        
    Returns:
        Detailed citation information or None if not found
    """
    try:
        # Get the citing and cited opinions
        citing_opinion = Neo4jOpinion.nodes.get(cluster_id=citing_id)
        cited_opinion = Neo4jOpinion.nodes.get(cluster_id=cited_id)
        
        # Get the relationship
        rel = citing_opinion.cites.relationship(cited_opinion)
        if not rel:
            logger.warning(f"No citation relationship found between {citing_id} and {cited_id}")
            return None
        
        # Create citation detail
        citation = CitationDetail.from_neo4j_rel(rel, citing_opinion, cited_opinion)
        
        return citation
    
    except Neo4jOpinion.DoesNotExist:
        logger.warning(f"Opinion not found: citing_id={citing_id} or cited_id={cited_id}")
        return None
    
    except Exception as e:
        logger.error(f"Error getting citation detail: {str(e)}")
        return None

def get_citation_stats(
    neo4j_session,
    court_id: Optional[str] = None,
    year: Optional[int] = None
) -> CitationStats:
    """
    Get citation statistics.
    
    Args:
        neo4j_session: Neo4j session
        court_id: Filter by court ID
        year: Filter by year
        
    Returns:
        Citation statistics
    """
    try:
        # Build query filters
        query = {}
        
        if court_id:
            query["court_id"] = court_id
        
        if year:
            # This is a simplification - in a real implementation, we would need to filter by year
            # For now, we'll just use the court_id filter
            pass
        
        # Get all opinions matching the query
        if query:
            opinions = Neo4jOpinion.nodes.filter(**query)
        else:
            opinions = Neo4jOpinion.nodes.all()
        
        # Calculate statistics
        total_opinions = len(opinions)
        total_citations = 0
        treatment_counts = {}
        relevance_distribution = {}
        section_distribution = {}
        
        # Process each opinion
        for opinion in opinions:
            # Get outgoing citations
            citations = opinion.cites.all()
            total_citations += len(citations)
            
            # Process each citation
            for cited in citations:
                rel = opinion.cites.relationship(cited)
                
                # Count by treatment
                treatment = rel.treatment
                if treatment:
                    treatment_counts[treatment] = treatment_counts.get(treatment, 0) + 1
                
                # Count by relevance
                relevance = rel.relevance
                if relevance:
                    relevance_distribution[relevance] = relevance_distribution.get(relevance, 0) + 1
                
                # Count by section
                section = rel.opinion_section
                if section:
                    section_distribution[section] = section_distribution.get(section, 0) + 1
        
        # Calculate average citations per opinion
        avg_citations_per_opinion = total_citations / total_opinions if total_opinions > 0 else 0
        
        # Create stats
        stats = CitationStats(
            total_citations=total_citations,
            total_opinions=total_opinions,
            avg_citations_per_opinion=avg_citations_per_opinion,
            treatment_counts=treatment_counts,
            relevance_distribution=relevance_distribution,
            section_distribution=section_distribution,
            metadata={
                "court_id": court_id,
                "year": year
            }
        )
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting citation stats: {str(e)}")
        # Return default stats on error
        return CitationStats(
            total_citations=0,
            total_opinions=0,
            avg_citations_per_opinion=0.0,
            treatment_counts={},
            relevance_distribution={},
            section_distribution={}
        )

def get_influential_citations(
    neo4j_session,
    court_id: Optional[str] = None,
    limit: int = 20
) -> List[CitationDetail]:
    """
    Get the most influential citations based on citation count.
    
    Args:
        neo4j_session: Neo4j session
        court_id: Filter by court ID
        limit: Maximum number of results to return
        
    Returns:
        List of influential citations
    """
    try:
        # Build query filters
        query = {}
        
        if court_id:
            query["court_id"] = court_id
        
        # Get all opinions matching the query
        if query:
            opinions = Neo4jOpinion.nodes.filter(**query)
        else:
            opinions = Neo4jOpinion.nodes.all()
        
        # Calculate citation counts for each opinion
        citation_counts = {}
        for opinion in opinions:
            citation_count = len(opinion.cited_by.all())
            citation_counts[opinion.cluster_id] = (opinion, citation_count)
        
        # Sort by citation count and take top 'limit'
        top_cited = sorted(citation_counts.items(), key=lambda x: x[1][1], reverse=True)[:limit]
        
        # Get the most recent citation for each top cited opinion
        influential_citations = []
        for _, (cited_opinion, citation_count) in top_cited:
            # Get the most recent citing opinion
            citing_opinions = cited_opinion.cited_by.order_by("-date_filed")
            
            if citing_opinions:
                recent_citing = citing_opinions[0]
                
                # Get the relationship
                rel = recent_citing.cites.relationship(cited_opinion)
                
                if rel:
                    # Create citation detail
                    citation = CitationDetail.from_neo4j_rel(rel, recent_citing, cited_opinion)
                    influential_citations.append(citation)
        
        return influential_citations
    
    except Exception as e:
        logger.error(f"Error getting influential citations: {str(e)}")
        return []
