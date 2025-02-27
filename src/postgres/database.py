"""
PostgreSQL database connection and session management.

This module provides the core database functionality for the application,
including connection management, session creation, and utility functions.
"""
import os
import logging
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, Engine, Column, Integer, String, and_, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Import eyecite for citation parsing
try:
    from eyecite import get_citations
    from eyecite.resolve import resolve_citations
    EYECITE_AVAILABLE = True
except ImportError:
    EYECITE_AVAILABLE = False

# Configure logger
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
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base = declarative_base()

# Legacy name for backward compatibility
Session = SessionLocal

def get_engine() -> Engine:
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
    return SessionLocal

def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.

    Yields:
        Session: A SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    """
    Context manager to handle database sessions.

    Yields:
        Session: A SQLAlchemy session
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def init_db():
    """
    Initialize the database by creating all tables.

    Note: In production, use alembic for migrations instead.
    """
    # Import models here to avoid circular imports
    from .models import Base
    Base.metadata.create_all(bind=engine)

def verify_connection() -> bool:
    """
    Verify database connection.

    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with get_db_session() as session:
            result = session.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"Successfully connected to PostgreSQL. Version: {version}")
            return True
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return False

# Citation lookup utility class and functions
class Citation(Base):
    """Model representing citation records in the database."""
    __tablename__ = "search_citation"

    id = Column(Integer, primary_key=True)
    volume = Column(String)
    reporter = Column(String)
    page = Column(String)
    type = Column(Integer)
    cluster_id = Column(Integer)

    def __repr__(self):
        return f"<Citation(volume={self.volume}, reporter={self.reporter}, page={self.page})>"

def find_cluster_id(citation_string: str) -> Optional[int]:
    """
    Find cluster_id for a given citation string using eyecite for parsing.
    
    Args:
        citation_string: Citation text to look up
        
    Returns:
        Optional[int]: Cluster ID if found, None otherwise
    """
    if not citation_string or not EYECITE_AVAILABLE:
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