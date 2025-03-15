import logging
import traceback
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from neomodel import adb
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.database import get_db

router = APIRouter(
    prefix="/api/stats",
    tags=["stats"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)


@router.get("")
async def get_stats(
    db: Session = Depends(get_db),
    #neo4j_session = Depends(get_neo4j)
):
    """
    Get statistics about citations and opinions.
    
    Returns:
        dict: Various statistics about the database
    """
    try:
        # Get Neo4j stats
        neo4j_stats = await get_neo4j_stats()
        
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

async def get_neo4j_stats() -> Dict[str, Any]:
    """
    Get statistics from Neo4j database.
    
    Args:
        neo4j_session: Neo4j session
        
    Returns:
        dict: Statistics from Neo4j
    """
    try:
        # We know the correct labels and properties from the logs
        opinion_label = ":`LegalDocument`:`Opinion`"
        
        # Count total citations
        citation_result = await adb.cypher_query("""
            MATCH ()-[r:CITES]->() 
            RETURN COUNT(r) as citation_count
        """)
        citation_count = citation_result[0][0] if citation_result and citation_result[0] else 0
        
        # Count total opinions
        opinion_result = await adb.cypher_query(f"""
            MATCH (o{opinion_label}) 
            RETURN COUNT(o) as opinion_count
        """)
        opinion_count = opinion_result[0][0] if opinion_result and opinion_result[0] else 0
        
        # Count opinions with AI summaries
        ai_summary_result = await adb.cypher_query(f"""
            MATCH (o{opinion_label})
            WHERE o.ai_summary IS NOT NULL AND o.ai_summary <> ''
            RETURN COUNT(o) as ai_summary_count
        """)
        ai_summary_count = ai_summary_result[0][0] if ai_summary_result and ai_summary_result[0] else 0
        logger.info(f"ai_summary_count: {ai_summary_count}")
        # Count citations by document type
        citation_types_result = await adb.cypher_query("""
            MATCH (source)-[r:CITES]->(target)
            WITH labels(target) AS target_labels
            WITH 
                CASE 
                    WHEN 'LawReview' IN target_labels THEN 'law_review'
                    WHEN 'Opinion' IN target_labels THEN 'judicial_opinion'
                    WHEN 'StatutesCodesRegulation' IN target_labels THEN 'statutes_codes_regulations'
                    WHEN 'ConstitutionalDocument' IN target_labels THEN 'constitution'
                    WHEN 'AdministrativeAgencyRuling' IN target_labels THEN 'administrative_agency_ruling'
                    WHEN 'CongressionalReport' IN target_labels THEN 'congressional_report'
                    WHEN 'ExternalSubmission' IN target_labels THEN 'external_submission'
                    WHEN 'ElectronicResource' IN target_labels THEN 'electronic_resource'
                    WHEN 'LegalDictionary' IN target_labels THEN 'legal_dictionary'
                    ELSE 'other'
                END AS citation_type
            RETURN citation_type, COUNT(*) as count
            ORDER BY count DESC
        """)
        
        result_rows, result_cols = citation_types_result
        citation_types = {}
        
        if "citation_type" in result_cols and "count" in result_cols:
            type_index = result_cols.index("citation_type")
            count_index = result_cols.index("count")
            for row in result_rows:
                citation_types[row[type_index]] = row[count_index]
        
        return {
            "citation_count": citation_count,
            "opinion_count": opinion_count,
            "ai_summary_count": ai_summary_count,
            "citation_types": citation_types
        }
    except Exception as e:
        logger.error(f"Error getting Neo4j stats: {e}, {traceback.format_exc()}")
        return {
            "error": str(e),
            "citation_count": 0,
            "opinion_count": 0,
            "ai_summary_count": 0,
            "citation_types": {}
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
        
        # Default values
        total_opinions = 0
        opinions_by_jurisdiction = {}
        
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
            "opinions_by_jurisdiction": opinions_by_jurisdiction
        }
    except Exception as e:
        logger.error(f"Error getting PostgreSQL stats: {e}")
        return {
            "error": str(e),
            "total_opinions": 0,
            "opinions_by_jurisdiction": {}
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
            
        return (neo4j_opinion_count[0] / postgres_opinion_count) * 100
    except Exception as e:
        logger.error(f"Error calculating coverage: {e}, {neo4j_stats}, {postgres_stats}")
        return 0.0

