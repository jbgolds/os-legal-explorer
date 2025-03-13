from typing import Generator

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.neo4j_db.neomodel_loader import neomodel_loader
from neomodel import adb
# Import database connections from updated modules
from src.postgres.database import get_engine, get_session_factory

# Load environment variables
load_dotenv()

# Get existing database connections
engine = get_engine()
SessionLocal = get_session_factory(engine)



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
async def get_neo4j():
    """Get Neo4j session."""
    #try:
        # Create a new session for each request
    async with adb.session() as session:
        yield session
    # finally:
    #     await session.close()


# Verify database connections
def verify_connections() -> dict:
    """
    Verify connections to both databases.

    Returns:
        dict: Connection status for each database
    """
    status = {"postgresql": False, "neo4j": False}

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
        with adb.session() as session:
            session.run("RETURN 1")
            status["neo4j"] = True
    except Exception as e:
        print(f"Neo4j connection error: {e}")

    return status

