from neo4j import GraphDatabase
from typing import List, Dict, Optional, Any
import logging
from datetime import date

from ..models.stats import (
    NetworkStats, 
    CourtStats, 
    TimelineStats,
    YearlyStats,
    TopCitedOpinion,
    TopCitedOpinions
)

logger = logging.getLogger(__name__)

def get_network_stats(neo4j_session) -> NetworkStats:
    """
    Get overall statistics about the citation network.
    
    Args:
        neo4j_session: Neo4j session
        
    Returns:
        Overall network statistics
    """
    query = """
    // Get basic network metrics
    MATCH (o:Opinion)
    OPTIONAL MATCH (o)-[r:CITES]->()
    
    WITH 
        count(DISTINCT o) as total_nodes,
        count(r) as total_edges
    
    // Calculate network density and average degree
    WITH 
        total_nodes,
        total_edges,
        CASE 
            WHEN total_nodes > 1 
            THEN total_edges / (total_nodes * (total_nodes - 1))
            ELSE 0 
        END as network_density,
        CASE 
            WHEN total_nodes > 0 
            THEN toFloat(total_edges) / total_nodes
            ELSE 0 
        END as avg_degree
    
    // Get max degree
    CALL {
        MATCH (o:Opinion)
        OPTIONAL MATCH (o)-[r:CITES]-()
        WITH o, count(r) as degree
        RETURN max(degree) as max_degree
    }
    
    // Get connected components (simplified approximation)
    CALL {
        MATCH (o:Opinion)
        WHERE NOT (o)-[:CITES]-()
        RETURN count(o) as isolated_nodes
    }
    
    // Return network statistics
    RETURN 
        total_nodes,
        total_edges,
        network_density,
        avg_degree,
        max_degree,
        isolated_nodes,
        total_nodes - isolated_nodes as largest_component_size
    """
    
    try:
        # Execute the query
        result = neo4j_session.run(query)
        record = result.single()
        
        if not record:
            # Return default stats if no data
            return NetworkStats(
                total_nodes=0,
                total_edges=0,
                network_density=0.0,
                avg_degree=0.0,
                max_degree=0,
                connected_components=0,
                largest_component_size=0
            )
        
        # Create network stats
        stats = NetworkStats(
            total_nodes=record["total_nodes"],
            total_edges=record["total_edges"],
            network_density=record["network_density"],
            avg_degree=record["avg_degree"],
            max_degree=record["max_degree"],
            # These are approximations since full graph algorithms would be expensive
            connected_components=1 + record["isolated_nodes"],  # Simplified
            largest_component_size=record["largest_component_size"],
            # These would require more complex graph algorithms
            avg_path_length=None,
            diameter=None,
            clustering_coefficient=None,
            metadata={
                "timestamp": date.today().isoformat()
            }
        )
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting network stats: {str(e)}")
        # Return default stats on error
        return NetworkStats(
            total_nodes=0,
            total_edges=0,
            network_density=0.0,
            avg_degree=0.0,
            max_degree=0,
            connected_components=0,
            largest_component_size=0
        )

def get_court_stats(neo4j_session, limit: int = 10) -> List[CourtStats]:
    """
    Get citation statistics by court.
    
    Args:
        neo4j_session: Neo4j session
        limit: Maximum number of courts to return
        
    Returns:
        List of court statistics
    """
    query = """
    // Get all courts with their opinion counts
    MATCH (o:Opinion)
    WHERE o._court_id IS NOT NULL
    WITH o._court_id as court_id, o.court_name as court_name, count(o) as opinion_count
    ORDER BY opinion_count DESC
    LIMIT $limit
    
    // Get citation counts for each court
    MATCH (o:Opinion)-[r:CITES]->()
    WHERE o._court_id = court_id
    WITH court_id, court_name, opinion_count, count(r) as citation_count
    
    // Get cited-by counts for each court
    MATCH (o:Opinion)<-[r:CITES]-()
    WHERE o._court_id = court_id
    WITH court_id, court_name, opinion_count, citation_count, count(r) as cited_by_count
    
    // Get self-citation counts
    MATCH (o1:Opinion)-[r:CITES]->(o2:Opinion)
    WHERE o1._court_id = court_id AND o2._court_id = court_id
    WITH 
        court_id, 
        court_name, 
        opinion_count, 
        citation_count, 
        cited_by_count,
        count(r) as self_citation_count
    
    // Calculate self-citation ratio
    WITH 
        court_id, 
        court_name, 
        opinion_count, 
        citation_count, 
        cited_by_count,
        CASE 
            WHEN citation_count > 0 
            THEN self_citation_count / toFloat(citation_count)
            ELSE 0 
        END as self_citation_ratio,
        CASE 
            WHEN opinion_count > 0 
            THEN citation_count / toFloat(opinion_count)
            ELSE 0 
        END as avg_citations_per_opinion
    
    // Get top cited courts
    CALL {
        WITH court_id
        MATCH (o1:Opinion)-[r:CITES]->(o2:Opinion)
        WHERE o1._court_id = court_id AND o2._court_id <> court_id
        WITH o2._court_id as cited_court_id, o2.court_name as cited_court_name, count(r) as count
        ORDER BY count DESC
        LIMIT 5
        RETURN collect({court_id: cited_court_id, court_name: cited_court_name, count: count}) as top_cited_courts
    }
    
    // Get top citing courts
    CALL {
        WITH court_id
        MATCH (o1:Opinion)-[r:CITES]->(o2:Opinion)
        WHERE o1._court_id <> court_id AND o2._court_id = court_id
        WITH o1._court_id as citing_court_id, o1.court_name as citing_court_name, count(r) as count
        ORDER BY count DESC
        LIMIT 5
        RETURN collect({court_id: citing_court_id, court_name: citing_court_name, count: count}) as top_citing_courts
    }
    
    // Return court statistics
    RETURN 
        court_id,
        court_name,
        opinion_count,
        citation_count,
        cited_by_count,
        self_citation_ratio,
        avg_citations_per_opinion,
        top_cited_courts,
        top_citing_courts
    ORDER BY opinion_count DESC
    """
    
    try:
        # Execute the query
        result = neo4j_session.run(query, {"limit": limit})
        records = result.records()
        
        court_stats_list = []
        for record in records:
            # Create court stats
            stats = CourtStats(
                court_id=record["court_id"],
                court_name=record["court_name"] or f"Court {record['court_id']}",
                opinion_count=record["opinion_count"],
                citation_count=record["citation_count"],
                cited_by_count=record["cited_by_count"],
                self_citation_ratio=record["self_citation_ratio"],
                avg_citations_per_opinion=record["avg_citations_per_opinion"],
                top_cited_courts=record["top_cited_courts"],
                top_citing_courts=record["top_citing_courts"],
                metadata={
                    "timestamp": date.today().isoformat()
                }
            )
            
            court_stats_list.append(stats)
        
        return court_stats_list
    
    except Exception as e:
        logger.error(f"Error getting court stats: {str(e)}")
        return []

def get_timeline_stats(
    neo4j_session,
    court_id: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> TimelineStats:
    """
    Get citation statistics over time.
    
    Args:
        neo4j_session: Neo4j session
        court_id: Filter by court ID
        start_year: Start year for timeline
        end_year: End year for timeline
        
    Returns:
        Timeline statistics
    """
    # Build the query based on filters
    filters = []
    params = {}
    
    if court_id:
        filters.append("o._court_id = $court_id")
        params["court_id"] = court_id
    
    filter_clause = " AND ".join(filters)
    if filter_clause:
        filter_clause = f"WHERE {filter_clause}"
    
    query = f"""
    // Get year range
    MATCH (o:Opinion)
    WHERE o.soc_date_filed IS NOT NULL
    {filter_clause}
    WITH 
        min(date.year(o.soc_date_filed)) as min_year,
        max(date.year(o.soc_date_filed)) as max_year
    
    // Apply year range filters if provided
    WITH 
        CASE WHEN $start_year IS NOT NULL AND $start_year > min_year THEN $start_year ELSE min_year END as start_year,
        CASE WHEN $end_year IS NOT NULL AND $end_year < max_year THEN $end_year ELSE max_year END as end_year
    
    // Generate years in range
    UNWIND range(start_year, end_year) as year
    
    // Get opinion counts by year
    CALL {{
        MATCH (o:Opinion)
        WHERE date.year(o.soc_date_filed) = year
        {filter_clause}
        RETURN count(o) as opinion_count
    }}
    
    // Get citation counts by year (outgoing)
    CALL {{
        MATCH (o:Opinion)-[r:CITES]->()
        WHERE date.year(o.soc_date_filed) = year
        {filter_clause}
        RETURN count(r) as citation_count
    }}
    
    // Get cited-by counts by year (incoming, regardless of when the citing opinion was published)
    CALL {{
        MATCH (o:Opinion)<-[r:CITES]-()
        WHERE date.year(o.soc_date_filed) = year
        {filter_clause}
        RETURN count(r) as cited_by_count
    }}
    
    // Calculate average citations per opinion
    WITH 
        year,
        opinion_count,
        citation_count,
        cited_by_count,
        CASE 
            WHEN opinion_count > 0 
            THEN citation_count / toFloat(opinion_count)
            ELSE 0 
        END as avg_citations_per_opinion
    
    // Return yearly statistics
    RETURN 
        collect({{
            year: year,
            opinion_count: opinion_count,
            citation_count: citation_count,
            cited_by_count: cited_by_count,
            avg_citations_per_opinion: avg_citations_per_opinion
        }}) as yearly_stats,
        min(year) as start_year,
        max(year) as end_year,
        count(year) as total_years
    """
    
    try:
        # Execute the query
        result = neo4j_session.run(query, {
            "court_id": court_id,
            "start_year": start_year,
            "end_year": end_year
        })
        record = result.single()
        
        if not record:
            # Return default stats if no data
            return TimelineStats(
                yearly_stats=[],
                total_years=0,
                start_year=start_year or 0,
                end_year=end_year or 0
            )
        
        # Process yearly stats
        yearly_stats = []
        for item in record["yearly_stats"]:
            stats = YearlyStats(
                year=item["year"],
                opinion_count=item["opinion_count"],
                citation_count=item["citation_count"],
                cited_by_count=item["cited_by_count"],
                avg_citations_per_opinion=item["avg_citations_per_opinion"],
                metadata={}
            )
            yearly_stats.append(stats)
        
        # Create timeline stats
        timeline_stats = TimelineStats(
            yearly_stats=yearly_stats,
            total_years=record["total_years"],
            start_year=record["start_year"],
            end_year=record["end_year"]
        )
        
        return timeline_stats
    
    except Exception as e:
        logger.error(f"Error getting timeline stats: {str(e)}")
        # Return default stats on error
        return TimelineStats(
            yearly_stats=[],
            total_years=0,
            start_year=start_year or 0,
            end_year=end_year or 0
        )

def get_top_cited_opinions(
    neo4j_session,
    court_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 20
) -> TopCitedOpinions:
    """
    Get the most cited opinions.
    
    Args:
        neo4j_session: Neo4j session
        court_id: Filter by court ID
        start_date: Filter by date range (start)
        end_date: Filter by date range (end)
        limit: Maximum number of opinions to return
        
    Returns:
        List of top cited opinions
    """
    # Build the query based on filters
    filters = []
    params = {"limit": limit}
    
    if court_id:
        filters.append("o._court_id = $court_id")
        params["court_id"] = court_id
    
    if start_date:
        filters.append("o.soc_date_filed >= $start_date")
        params["start_date"] = start_date
    
    if end_date:
        filters.append("o.soc_date_filed <= $end_date")
        params["end_date"] = end_date
    
    filter_clause = " AND ".join(filters)
    if filter_clause:
        filter_clause = f"WHERE {filter_clause}"
    
    query = f"""
    // Get total opinion count
    MATCH (o:Opinion)
    {filter_clause}
    WITH count(o) as total_count
    
    // Get top cited opinions
    MATCH (o:Opinion)<-[r:CITES]-()
    {filter_clause}
    WITH o, count(r) as citation_count, total_count
    ORDER BY citation_count DESC
    LIMIT $limit
    
    // Return top cited opinions
    RETURN 
        collect({{
            cluster_id: o.cluster_id,
            case_name: o.case_name,
            court_id: o._court_id,
            court_name: o.court_name,
            date_filed: o.soc_date_filed,
            citation_count: citation_count
        }}) as opinions,
        total_count
    """
    
    try:
        # Execute the query
        result = neo4j_session.run(query, params)
        record = result.single()
        
        if not record:
            # Return default stats if no data
            return TopCitedOpinions(
                opinions=[],
                total_count=0,
                metadata={}
            )
        
        # Process opinions
        opinions = []
        for item in record["opinions"]:
            opinion = TopCitedOpinion(
                cluster_id=item["cluster_id"],
                case_name=item["case_name"] or f"Opinion {item['cluster_id']}",
                court_id=item["court_id"] or "unknown",
                court_name=item["court_name"] or "Unknown Court",
                date_filed=item["date_filed"],
                citation_count=item["citation_count"],
                metadata={}
            )
            opinions.append(opinion)
        
        # Create top cited opinions
        top_cited = TopCitedOpinions(
            opinions=opinions,
            total_count=record["total_count"],
            metadata={
                "court_id": court_id,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "limit": limit
            }
        )
        
        return top_cited
    
    except Exception as e:
        logger.error(f"Error getting top cited opinions: {str(e)}")
        # Return default stats on error
        return TopCitedOpinions(
            opinions=[],
            total_count=0,
            metadata={}
        )

def get_citation_distribution(neo4j_session, bins: int = 10) -> Dict[str, Any]:
    """
    Get the distribution of citation counts.
    
    Args:
        neo4j_session: Neo4j session
        bins: Number of bins for the distribution
        
    Returns:
        Citation count distribution
    """
    query = """
    // Get citation counts for all opinions
    MATCH (o:Opinion)<-[r:CITES]-()
    WITH o, count(r) as citation_count
    
    // Get max citation count for binning
    WITH collect(citation_count) as counts, max(citation_count) as max_count
    
    // Create bins
    WITH counts, max_count, $bins as num_bins
    UNWIND counts as count
    WITH count, max_count, num_bins, floor(count * num_bins / (max_count + 1)) as bin
    
    // Count opinions in each bin
    WITH bin, count(bin) as bin_count, max_count, num_bins
    ORDER BY bin
    
    // Calculate bin ranges
    WITH 
        bin, 
        bin_count, 
        max_count, 
        num_bins,
        floor(bin * (max_count + 1) / num_bins) as bin_start,
        floor((bin + 1) * (max_count + 1) / num_bins) - 1 as bin_end
    
    // Return distribution
    RETURN 
        collect({
            bin: bin,
            bin_start: bin_start,
            bin_end: bin_end,
            count: bin_count,
            label: CASE 
                WHEN bin_start = bin_end THEN toString(bin_start)
                ELSE bin_start + '-' + bin_end
            END
        }) as distribution,
        max_count
    """
    
    try:
        # Execute the query
        result = neo4j_session.run(query, {"bins": bins})
        record = result.single()
        
        if not record:
            # Return default distribution if no data
            return {
                "distribution": [],
                "max_count": 0,
                "bins": bins
            }
        
        # Return distribution
        return {
            "distribution": record["distribution"],
            "max_count": record["max_count"],
            "bins": bins
        }
    
    except Exception as e:
        logger.error(f"Error getting citation distribution: {str(e)}")
        # Return default distribution on error
        return {
            "distribution": [],
            "max_count": 0,
            "bins": bins
        }
