import os
from typing import Generator
from sqlalchemy.orm import Session
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Import database connections from updated modules
from src.postgres.database import get_engine, get_session_factory
from src.neo4j.neomodel_loader import get_neo4j_driver

# Load environment variables
load_dotenv()

# Get existing database connections
engine = get_engine()
SessionLocal = get_session_factory(engine)
neo4j_driver = get_neo4j_driver()

# Dependency to get DB session
def get_db() -> Generator[Session, None, None]:
    """
    Get a PostgreSQL database session.
    
    Yields:
        Session: SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get Neo4j session
def get_neo4j() -> Generator[GraphDatabase.session, None, None]:
    """
    Get a Neo4j database session.
    
    Yields:
        GraphDatabase.session: Neo4j session
    """
    session = neo4j_driver.session()
    try:
        yield session
    finally:
        session.close()

# Verify database connections
def verify_connections() -> dict:
    """
    Verify connections to both databases.
    
    Returns:
        dict: Connection status for each database
    """
    status = {
        "postgresql": False,
        "neo4j": False
    }
    
    # Check PostgreSQL
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        status["postgresql"] = True
        db.close()
    except Exception as e:
        print(f"PostgreSQL connection error: {e}")
    
    # Check Neo4j
    try:
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
            status["neo4j"] = True
    except Exception as e:
        print(f"Neo4j connection error: {e}")
    
    return status

# Close connections on application shutdown
def close_connections():
    """Close all database connections."""
    neo4j_driver.close()
