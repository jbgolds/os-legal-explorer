"""
PostgreSQL database connection and session management.

This module provides the core database functionality for the application,
including connection management, session creation, and utility functions.
"""

import os
import logging
from typing import Generator, Optional
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import create_engine, Engine, Column, Integer, String, and_, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database credentials from environment variables
DB_USER = os.environ["POSTGRES_USER"]
DB_PASSWORD = os.environ["POSTGRES_PASSWORD"]
DB_HOST = os.environ["POSTGRES_HOST"]
DB_PORT = os.environ["POSTGRES_PORT"]
DB_NAME = os.environ["POSTGRES_DB"]

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
