import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from typing import Dict, Any

from ..database import get_db, get_neo4j

router = APIRouter(
    prefix="/api/stats",
    tags=["stats"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.get("/")
@router.get("")
async def get_stats(
    db: Session = Depends(get_db),
    neo4j_session = Depends(get_neo4j)
):
    """
    Get statistics about citations and opinions.
    
    Returns:
        dict: Various statistics about the database
    """
    try:
        # Get Neo4j stats
        neo4j_stats = await get_neo4j_stats(neo4j_session)
        
        # Get PostgreSQL stats
        postgres_stats = await get_postgres_stats(db)
        
        # Combine stats
        return {
            "neo4j": neo4j_stats,
            "postgres": postgres_stats,
            "comparison": {
                "coverage_percentage": calculate_coverage(neo4j_stats, postgres_stats)
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

async def get_neo4j_stats(neo4j_session) -> Dict[str, Any]:
    """
    Get statistics from Neo4j database.
    
    Args:
        neo4j_session: Neo4j session
        
    Returns:
        dict: Statistics from Neo4j
    """
    try:
        # First, let's check what labels and properties are available
        schema_result = neo4j_session.run("""
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType, propertyName
            RETURN nodeType, collect(propertyName) as properties
        """)
        
        schema = {record["nodeType"]: record["properties"] for record in schema_result}
        logger.info(f"Neo4j schema: {schema}")
        
        # Count total citations
        citation_result = neo4j_session.run("""
            MATCH ()-[r:CITES]->() 
            RETURN COUNT(r) as citation_count
        """)
        citation_count = citation_result.single()["citation_count"]
        
        # Count total opinions
        opinion_result = neo4j_session.run("""
            MATCH (o:Opinion) 
            RETURN COUNT(o) as opinion_count
        """)
        opinion_count = opinion_result.single()["opinion_count"]
        
        # Get citation distribution by year if the property exists
        citations_by_year = {}
        if "Opinion" in schema or ":`LegalDocument`:`Opinion`" in schema:
            # Find the correct node label
            opinion_label = "Opinion"
            for label in schema:
                if "Opinion" in label:
                    opinion_label = label
                    break
            
            # Check if date_filed property exists
            date_property = None
            if opinion_label in schema:
                available_props = schema[opinion_label]
                if "date_filed" in available_props:
                    date_property = "date_filed"
                elif "created_at" in available_props:
                    date_property = "created_at"
                elif "updated_at" in available_props:
                    date_property = "updated_at"
            
            if date_property:
                # Try to extract year from the date property
                # First, check the format of the date property by getting a sample
                sample_query = f"""
                    MATCH (o:{opinion_label.replace(':`', '').replace('`:', '').replace('`', '')})
                    WHERE o.{date_property} IS NOT NULL
                    RETURN o.{date_property} as date_value
                    LIMIT 1
                """
                
                try:
                    sample_result = neo4j_session.run(sample_query)
                    sample_record = sample_result.single()
                    
                    if sample_record:
                        date_value = sample_record["date_value"]
                        logger.info(f"Sample date value: {date_value} (type: {type(date_value).__name__})")
                        
                        # Based on the date format, construct the appropriate query
                        year_query = ""
                        if isinstance(date_value, str):
                            if "-" in date_value:  # ISO format like "2020-01-01"
                                year_query = f"""
                                    MATCH (o:{opinion_label.replace(':`', '').replace('`:', '').replace('`', '')})
                                    WHERE o.{date_property} IS NOT NULL
                                    WITH o, SUBSTRING(o.{date_property}, 0, 4) as year
                                    WHERE year =~ '[0-9]{{4}}'
                                    RETURN year, COUNT(o) as count
                                    ORDER BY year
                                """
                            elif "/" in date_value:  # Format like "01/01/2020"
                                year_query = f"""
                                    MATCH (o:{opinion_label.replace(':`', '').replace('`:', '').replace('`', '')})
                                    WHERE o.{date_property} IS NOT NULL
                                    WITH o, SUBSTRING(o.{date_property}, 6, 4) as year
                                    WHERE year =~ '[0-9]{{4}}'
                                    RETURN year, COUNT(o) as count
                                    ORDER BY year
                                """
                        else:
                            # For non-string dates, try a generic approach
                            year_query = f"""
                                MATCH (o:{opinion_label.replace(':`', '').replace('`:', '').replace('`', '')})
                                WHERE o.{date_property} IS NOT NULL
                                WITH o, toString(o.{date_property}) as date_str
                                WITH o, 
                                     CASE 
                                         WHEN date_str CONTAINS '-' THEN SUBSTRING(date_str, 0, 4)
                                         WHEN date_str CONTAINS '/' THEN SUBSTRING(date_str, 6, 4)
                                         ELSE '0000'
                                     END as year
                                WHERE year =~ '[0-9]{{4}}'
                                RETURN year, COUNT(o) as count
                                ORDER BY year
                            """
                        
                        if year_query:
                            logger.info(f"Using year query: {year_query}")
                            year_result = neo4j_session.run(year_query)
                            citations_by_year = {record["year"]: record["count"] for record in year_result}
                except Exception as e:
                    logger.error(f"Error getting sample date: {e}")
        
        # Sort the citations by year
        if citations_by_year:
            citations_by_year = dict(sorted(citations_by_year.items()))
        
        # Get top 10 most cited opinions
        top_cited = []
        if "Opinion" in schema or any("Opinion" in label for label in schema):
            # Find the correct node label
            opinion_label = "Opinion"
            for label in schema:
                if "Opinion" in label:
                    opinion_label = label
                    break
            
            # Check what properties are available for Opinion nodes
            available_props = schema.get(opinion_label, [])
            id_property = "id"
            name_property = "case_name"
            
            # Try to find appropriate properties for id and name
            if "primary_id" in available_props:
                id_property = "primary_id"
            elif "cluster_id" in available_props:
                id_property = "cluster_id"
            elif "id" in available_props:
                id_property = "id"
                
            if "case_name" in available_props:
                name_property = "case_name"
            elif "name" in available_props:
                name_property = "name"
            elif "title" in available_props:
                name_property = "title"
            
            # Use the identified properties in the query
            top_cited_query = f"""
                MATCH (o:{opinion_label.replace(':`', '').replace('`:', '').replace('`', '')})<-[r:CITES]-()
                RETURN o.{id_property} as cluster_id, o.{name_property} as case_name, COUNT(r) as citation_count
                ORDER BY citation_count DESC
                LIMIT 10
            """
            
            logger.info(f"Using top cited query: {top_cited_query}")
            
            try:
                top_cited_result = neo4j_session.run(top_cited_query)
                top_cited = [
                    {
                        "cluster_id": record["cluster_id"],
                        "case_name": record["case_name"] or "Unknown Case",
                        "citation_count": record["citation_count"]
                    }
                    for record in top_cited_result
                ]
            except Exception as e:
                logger.error(f"Error getting top cited opinions: {e}")
        
        return {
            "citation_count": citation_count,
            "opinion_count": opinion_count,
            "citations_by_year": citations_by_year,
            "top_cited_opinions": top_cited
        }
    except Exception as e:
        logger.error(f"Error getting Neo4j stats: {e}")
        return {
            "error": str(e),
            "citation_count": 0,
            "opinion_count": 0,
            "citations_by_year": {},
            "top_cited_opinions": []
        }

async def get_postgres_stats(db: Session) -> Dict[str, Any]:
    """
    Get statistics from PostgreSQL database.
    
    Args:
        db: SQLAlchemy session
        
    Returns:
        dict: Statistics from PostgreSQL
    """
    try:
        # First, let's check what tables are available
        tables_result = db.execute(
            text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
        )
        available_tables = [row[0] for row in tables_result]
        logger.info(f"PostgreSQL available tables: {available_tables}")
        
        # Default values
        total_opinions = 0
        opinions_by_jurisdiction = {}
        opinions_by_year = {}
        
        # Find the opinions table - it might be named differently
        opinion_table = None
        for table in available_tables:
            if 'opinion' in table.lower():
                opinion_table = table
                break
        
        if opinion_table:
            # Count total opinions
            total_opinions = db.execute(
                text(f"SELECT COUNT(*) FROM {opinion_table}")
            ).scalar()
            
            # Check if the table has a date_filed column
            columns_result = db.execute(
                text(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{opinion_table}'
                """)
            )
            available_columns = [row[0] for row in columns_result]
            logger.info(f"Columns in {opinion_table}: {available_columns}")
            
            # Get opinions by year if date_filed exists
            date_column = None
            for column in available_columns:
                if 'date' in column.lower() or 'filed' in column.lower():
                    date_column = column
                    break
            
            if date_column:
                year_result = db.execute(
                    text(f"""
                        SELECT EXTRACT(YEAR FROM {date_column}) as year, COUNT(*) as count
                        FROM {opinion_table}
                        WHERE {date_column} IS NOT NULL
                        GROUP BY year
                        ORDER BY year
                    """)
                )
                opinions_by_year = {
                    int(row[0]): row[1] for row in year_result
                }
            
            # Try to find jurisdiction information
            court_table = None
            for table in available_tables:
                if 'court' in table.lower():
                    court_table = table
                    break
            
            if court_table:
                # Check if the court table has a jurisdiction column
                court_columns_result = db.execute(
                    text(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = '{court_table}'
                    """)
                )
                court_columns = [row[0] for row in court_columns_result]
                
                jurisdiction_column = None
                for column in court_columns:
                    if 'jurisdiction' in column.lower():
                        jurisdiction_column = column
                        break
                
                if jurisdiction_column:
                    # Try to find a foreign key relationship between opinion and court tables
                    fk_result = db.execute(
                        text(f"""
                            SELECT kcu.column_name, ccu.table_name, ccu.column_name
                            FROM information_schema.table_constraints tc
                            JOIN information_schema.key_column_usage kcu
                              ON tc.constraint_name = kcu.constraint_name
                            JOIN information_schema.constraint_column_usage ccu
                              ON ccu.constraint_name = tc.constraint_name
                            WHERE tc.constraint_type = 'FOREIGN KEY'
                              AND tc.table_name = '{opinion_table}'
                              AND ccu.table_name = '{court_table}'
                        """)
                    )
                    
                    fk_columns = [(row[0], row[1], row[2]) for row in fk_result]
                    
                    if fk_columns:
                        # We found a foreign key relationship
                        opinion_fk_column, court_table, court_pk_column = fk_columns[0]
                        
                        jurisdiction_result = db.execute(
                            text(f"""
                                SELECT c.{jurisdiction_column}, COUNT(*) as count
                                FROM {opinion_table} o
                                JOIN {court_table} c ON o.{opinion_fk_column} = c.{court_pk_column}
                                GROUP BY c.{jurisdiction_column}
                                ORDER BY count DESC
                            """)
                        )
                        
                        opinions_by_jurisdiction = {
                            row[0]: row[1] for row in jurisdiction_result if row[0] is not None
                        }
                    else:
                        # No foreign key found, just count by court table
                        jurisdiction_result = db.execute(
                            text(f"""
                                SELECT {jurisdiction_column}, COUNT(*) as count
                                FROM {court_table}
                                GROUP BY {jurisdiction_column}
                                ORDER BY count DESC
                            """)
                        )
                        
                        opinions_by_jurisdiction = {
                            row[0]: row[1] for row in jurisdiction_result if row[0] is not None
                        }
        
        return {
            "total_opinions": total_opinions,
            "opinions_by_jurisdiction": opinions_by_jurisdiction,
            "opinions_by_year": opinions_by_year
        }
    except Exception as e:
        logger.error(f"Error getting PostgreSQL stats: {e}")
        return {
            "error": str(e),
            "total_opinions": 0,
            "opinions_by_jurisdiction": {},
            "opinions_by_year": {}
        }

def calculate_coverage(neo4j_stats: Dict[str, Any], postgres_stats: Dict[str, Any]) -> float:
    """
    Calculate the coverage percentage of opinions in Neo4j compared to PostgreSQL.
    
    Args:
        neo4j_stats: Statistics from Neo4j
        postgres_stats: Statistics from PostgreSQL
        
    Returns:
        float: Coverage percentage
    """
    try:
        neo4j_opinion_count = neo4j_stats.get("opinion_count", 0)
        postgres_opinion_count = postgres_stats.get("total_opinions", 0)
        
        if postgres_opinion_count == 0:
            return 0.0
            
        return (neo4j_opinion_count / postgres_opinion_count) * 100
    except Exception as e:
        logger.error(f"Error calculating coverage: {e}")
        return 0.0 