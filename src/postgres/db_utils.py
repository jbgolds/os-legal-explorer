from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
from sqlalchemy import and_
from eyecite import get_citations
from eyecite.resolve import resolve_citations
import logging
from typing import Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database credentials from environment variables
DB_USER = os.getenv("DB_USER", "courtlistener")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgrespassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "courtlistener")

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,
)

# Create session factory
Session = sessionmaker(bind=engine)

# Create base class for declarative models
Base = declarative_base()

def get_engine():
    """
    Get the SQLAlchemy engine instance.
    
    Returns:
        Engine: SQLAlchemy engine
    """
    return engine

def get_session_factory(engine_instance=None):
    """
    Get the SQLAlchemy session factory.
    
    Args:
        engine_instance: Optional engine instance to bind to the session factory
        
    Returns:
        sessionmaker: SQLAlchemy session factory
    """
    if engine_instance:
        return sessionmaker(autocommit=False, autoflush=False, bind=engine_instance)
    return Session


class Citation(Base):
    __tablename__ = "search_citation"

    id = Column(Integer, primary_key=True)
    volume = Column(String)
    reporter = Column(String)
    page = Column(String)
    type = Column(Integer)
    cluster_id = Column(Integer)

    def __repr__(self):
        return f"<Citation(volume={self.volume}, reporter={self.reporter}, page={self.page})>"


@contextmanager
def get_db_session():
    """Context manager for database sessions with automatic cleanup."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        session.close()


def find_cluster_id(citation_string: str) -> Optional[int]:
    """Find cluster_id for a given citation string using eyecite for parsing"""
    if not citation_string:
        return None

    try:
        # Extract citations using eyecite
        citations = get_citations(citation_string)
        if not citations:
            logger.debug(f"No citations found in: {citation_string}")
            return None

        # Resolve the citations
        resolved_citations = resolve_citations(citations)
        if not resolved_citations:
            logger.debug(f"Could not resolve citations in: {citation_string}")
            return None

        # Use the first resolved citation
        resolved = list(resolved_citations.keys())[0]
        resolved_citation = resolved.citation

        # Extract normalized components
        volume = str(resolved_citation.groups["volume"])
        reporter = resolved_citation.groups["reporter"]
        page = str(resolved_citation.groups["page"])

        if not volume or not reporter or not page:
            logger.warning(
                f"Invalid citation lookup: {citation_string}, missing volume, reporter, or page"
            )
            return None

        # Use database session with proper error handling
        with get_db_session() as session:
            # Query the database with resolved components
            result = (
                session.query(Citation)
                .filter(
                    and_(
                        Citation.volume == volume,
                        Citation.reporter == reporter,
                        Citation.page == page,
                    )
                )
                .first()
            )

            return result.cluster_id if result else None

    except Exception as e:
        logger.error(f"Error processing citation '{citation_string}': {str(e)}")
        return None


def test_database_connection():
    """Test the database connection and configuration."""
    try:
        with get_db_session() as session:
            result = session.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"Successfully connected to PostgreSQL. Version: {version}")
            return True
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return False


# def normalize_citation(citation_string):
#     """Normalize a citation string into components"""
#     parts = citation_string.strip().split()
#     if len(parts) >= 3:
#         volume = parts[0]
#         reporter = parts[1]  # .rstrip(".,")
#         page = parts[2]  # .rstrip(".,")
#         return volume, reporter, page
#     return None


# def get_case_name_from_cluster_id(cluster_id: int):
#     """Get case name from cluster_id using eyecite for parsing"""
#     try:
#         # Get database session
#         db = next(get_db())

#         # Query the database with resolved components
#         result = (
#             db.query(Citation)
#             .filter(
#                 and_(
#                     Citation.volume == volume,
#                     Citation.reporter == reporter,
#                     Citation.page == page,
#                 )
#             )
# # Example usage
# if __name__ == "__main__":
#     test_citations = [
#         "235 Kan. 195",
#         "347 U.S. 483",  # Brown v. Board of Education
#         "410 U.S. 113",  # Roe v. Wade
#     ]

#     for cite in test_citations:
#         cluster_id = find_cluster_id(cite)
#         print(f"Citation: {cite} -> Cluster ID: {cluster_id}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Test the connection
    test_database_connection()
