from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

from src.neo4j.models import Opinion as Neo4jOpinion, CitesRel
from src.llm_extraction.models import Citation, CitationAnalysis
from .db_utils import get_opinion_by_id, get_filtered_opinions

logger = logging.getLogger(__name__)

def opinion_to_node_dict(opinion: Neo4jOpinion, node_type: str = "opinion") -> dict:
    """Helper to convert a Neo4j Opinion to a node dictionary."""
    return {
        "id": opinion.cluster_id,
        "label": opinion.case_name or f"Opinion {opinion.cluster_id}",
        "type": node_type,
        "court_id": opinion.court_id,
        "date_filed": opinion.date_filed.isoformat() if opinion.date_filed else None,
        "citation_count": len(opinion.cited_by.all()) if hasattr(opinion, 'cited_by') else 0
    }

def get_citation_network(
    neo4j_session,  # Kept for API compatibility but not used
    cluster_id: Optional[int] = None,
    court_id: Optional[str] = None,
    depth: int = 1,
    limit: int = 100
) -> Dict[str, Any]:
    """Get a citation network centered around a specific opinion or court using neomodel."""
    depth = min(max(depth, 1), 3)  # Clamp depth between 1-3
    nodes = []
    edges = []
    
    try:
        if cluster_id:
            # Get center opinion using simplified db_utils
            center_opinion = get_opinion_by_id(cluster_id)
            if not center_opinion:
                return {"nodes": [], "edges": [], "metadata": {}}
                
            nodes.append(opinion_to_node_dict(center_opinion, "center"))
            
            # Get outgoing citations using relationship traversal
            cited_opinions = set()
            current_level = {center_opinion}
            
            for _ in range(depth):
                next_level = set()
                for opinion in current_level:
                    # Get direct citations limited by the limit parameter
                    new_cited = set(opinion.cites.all()[:limit // len(current_level)])
                    
                    for cited in new_cited:
                        if cited.cluster_id not in {n["id"] for n in nodes}:
                            nodes.append(opinion_to_node_dict(cited, "cited"))
                        
                        # Add edge using relationship properties
                        rel = opinion.cites.relationship(cited)
                        if rel:
                            edges.append({
                                "source": opinion.cluster_id,
                                "target": cited.cluster_id,
                                "treatment": rel.treatment,
                                "relevance": rel.relevance
                            })
                    
                    next_level.update(new_cited)
                cited_opinions.update(next_level)
                current_level = next_level
                
            # Get incoming citations similarly
            citing_opinions = set()
            current_level = {center_opinion}
            
            for _ in range(depth):
                next_level = set()
                for opinion in current_level:
                    new_citing = set(opinion.cited_by.all()[:limit // len(current_level)])
                    
                    for citing in new_citing:
                        if citing.cluster_id not in {n["id"] for n in nodes}:
                            nodes.append(opinion_to_node_dict(citing, "citing"))
                        
                        rel = citing.cites.relationship(opinion)
                        if rel:
                            edges.append({
                                "source": citing.cluster_id,
                                "target": opinion.cluster_id,
                                "treatment": rel.treatment,
                                "relevance": rel.relevance
                            })
                    
                    next_level.update(new_citing)
                citing_opinions.update(next_level)
                current_level = next_level

        elif court_id:
            # Get all opinions from this court using simplified db_utils
            court_opinions = get_filtered_opinions(
                court_id=court_id,
                limit=limit
            )
            
            for opinion in court_opinions:
                if opinion.cluster_id not in {n["id"] for n in nodes}:
                    nodes.append(opinion_to_node_dict(opinion, "court"))
                
                # Get direct citations and citing opinions
                for cited in opinion.cites.all()[:limit // len(court_opinions)]:
                    if cited.cluster_id not in {n["id"] for n in nodes}:
                        nodes.append(opinion_to_node_dict(cited))
                    
                    rel = opinion.cites.relationship(cited)
                    if rel:
                        edges.append({
                            "source": opinion.cluster_id,
                            "target": cited.cluster_id,
                            "treatment": rel.treatment,
                            "relevance": rel.relevance
                        })
                
                for citing in opinion.cited_by.all()[:limit // len(court_opinions)]:
                    if citing.cluster_id not in {n["id"] for n in nodes}:
                        nodes.append(opinion_to_node_dict(citing))
                    
                    rel = citing.cites.relationship(opinion)
                    if rel:
                        edges.append({
                            "source": citing.cluster_id,
                            "target": opinion.cluster_id,
                            "treatment": rel.treatment,
                            "relevance": rel.relevance
                        })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "center_id": cluster_id,
                "court_id": court_id,
                "depth": depth,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }
    except Neo4jOpinion.DoesNotExist:
        logger.warning(f"Opinion not found: cluster_id={cluster_id}")
        return {"nodes": [], "edges": [], "metadata": {}}
    except Exception as e:
        logger.error(f"Error getting citation network: {str(e)}")
        return {"nodes": [], "edges": [], "metadata": {}}

def get_citation_detail(
    neo4j_session,  # Kept for API compatibility but not used
    citing_id: int,
    cited_id: int
) -> Optional[Dict[str, Any]]:
    """Get citation details using neomodel relationship traversal."""
    try:
        # Use db_utils to get opinions by ID
        citing_opinion = get_opinion_by_id(citing_id)
        cited_opinion = get_opinion_by_id(cited_id)
        
        if not citing_opinion or not cited_opinion:
            return None
        
        rel = citing_opinion.cites.relationship(cited_opinion)
        if not rel:
            return None
        
        return {
            "citing_opinion": opinion_to_node_dict(citing_opinion),
            "cited_opinion": opinion_to_node_dict(cited_opinion),
            "citation": {
                "text": rel.citation_text,
                "page_number": rel.page_number,
                "treatment": rel.treatment,
                "relevance": rel.relevance,
                "reasoning": rel.reasoning,
                "section": rel.opinion_section,
                "source": rel.source
            }
        }
    except Exception as e:
        logger.error(f"Error getting citation detail: {str(e)}")
        return None

def get_citation_stats(
    neo4j_session,  # Kept for API compatibility but not used
    court_id: Optional[str] = None,
    year: Optional[int] = None
) -> Dict[str, Any]:
    """Get citation statistics using neomodel filters and relationship traversal."""
    try:
        # Build filter conditions
        filters = {}
        if court_id:
            filters["court_id"] = court_id
        if year:
            filters["date_filed__year"] = year
            
        # Use db_utils to get filtered opinions
        opinions = get_filtered_opinions(
            court_id=court_id,
            limit=None,  # No limit for stats calculation
            **filters
        )
        
        # Calculate stats using relationship traversal
        total_opinions = len(opinions)
        total_citations = 0
        treatment_counts = {}
        relevance_distribution = {}
        section_distribution = {}
        
        for opinion in opinions:
            citations = opinion.cites.all()
            total_citations += len(citations)
            
            for cited in citations:
                rel = opinion.cites.relationship(cited)
                if rel.treatment:
                    treatment_counts[rel.treatment] = treatment_counts.get(rel.treatment, 0) + 1
                if rel.relevance:
                    relevance_distribution[rel.relevance] = relevance_distribution.get(rel.relevance, 0) + 1
                if rel.opinion_section:
                    section_distribution[rel.opinion_section] = section_distribution.get(rel.opinion_section, 0) + 1
        
        return {
            "total_citations": total_citations,
            "total_opinions": total_opinions,
            "avg_citations_per_opinion": total_citations / total_opinions if total_opinions > 0 else 0,
            "treatment_counts": treatment_counts,
            "relevance_distribution": relevance_distribution,
            "section_distribution": section_distribution,
            "metadata": {"court_id": court_id, "year": year}
        }
    except Exception as e:
        logger.error(f"Error getting citation stats: {str(e)}")
        return {
            "total_citations": 0,
            "total_opinions": 0,
            "avg_citations_per_opinion": 0,
            "treatment_counts": {},
            "relevance_distribution": {},
            "section_distribution": {},
            "metadata": {}
        }

def get_influential_citations(
    neo4j_session,  # Kept for API compatibility but not used
    court_id: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Get influential citations using neomodel filters and relationship traversal."""
    try:
        # Use db_utils to get filtered opinions
        opinions = get_filtered_opinions(court_id=court_id, limit=None)
        
        # Get citation counts using relationship traversal
        citation_counts = {
            opinion.cluster_id: (opinion, len(opinion.cited_by.all()))
            for opinion in opinions
        }
        
        # Sort and get top cited
        top_cited = sorted(citation_counts.items(), key=lambda x: x[1][1], reverse=True)[:limit]
        
        influential = []
        for _, (cited_opinion, citation_count) in top_cited:
            # Use neomodel's order_by for citing opinions
            citing_opinions = cited_opinion.cited_by.order_by("-date_filed")
            
            if citing_opinions:
                recent_citing = citing_opinions[0]
                rel = recent_citing.cites.relationship(cited_opinion)
                
                if rel:
                    influential.append({
                        "cited_opinion": {
                            "cluster_id": cited_opinion.cluster_id,
                            "case_name": cited_opinion.case_name,
                            "citation_count": citation_count
                        },
                        "recent_citation": {
                            "citing_opinion": opinion_to_node_dict(recent_citing),
                            "treatment": rel.treatment,
                            "relevance": rel.relevance
                        }
                    })
        
        return influential
    
    except Exception as e:
        logger.error(f"Error getting influential citations: {str(e)}")
        return []
