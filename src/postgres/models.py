from datetime import datetime
from enum import Enum

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum as SQLAEnum, Text, Boolean
from sqlalchemy.orm import declarative_base, relationship

from src.llm_extraction.models import CitationType, CitationTreatment, OpinionSection, OpinionType

Base = declarative_base()

"""
NOTE THESE ARE NOT YET APPLIED TO OUR DATABASE AS THIS IS THE LAST THING TO BE DONE.

I BELIEVE WE CAN EVEN BUILD THIS FROM THE NEO4J OUTPUT
"""
class OpinionClusterExtraction(Base):
    __tablename__ = "opinion_cluster_extraction"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(
        Integer,
        unique=True,
        index=True,
        nullable=False,
        comment="The unique identifier for this opinion cluster",
    )
    date_filed = Column(
        DateTime, index=True, nullable=False, comment="The date the opinion was filed"
    )
    brief_summary = Column(
        String, nullable=False, comment="3-5 sentences describing the core holding"
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    citations = relationship("CitationExtraction", back_populates="opinion_cluster")
   #  opinion_text = relationship("OpinionText", back_populates="opinion_cluster", uselist=False)




class CitationExtraction(Base):
    __tablename__ = "citation_extraction"

    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_cluster_extraction_id = Column(
        Integer, ForeignKey("opinion_cluster_extraction.id"), index=True, nullable=False
    )

    # Citation section and type
    section = Column(SQLAEnum(OpinionSection), nullable=False)
    citation_type = Column(
        SQLAEnum(CitationType),
        nullable=False,
        comment="The type of legal document being cited",
    )

    # Citation metadata
    citation_text = Column(
        String, nullable=False, comment="The entirety of the extracted citation text"
    )
    page_number = Column(
        Integer, nullable=True, comment="The page number where this citation appears"
    )
    treatment = Column(
        SQLAEnum(CitationTreatment),
        nullable=True,
        comment="How this citation is treated (POSITIVE, NEGATIVE, etc)",
    )
    relevance = Column(Integer, nullable=True, comment="Relevance score between 1-4")
    reasoning = Column(
        String, nullable=True, comment="Detailed analysis explaining citation context"
    )

    # TODO, add in keys for other resolutions when we get there, i.e. for codes/laws/regulations.
    resolved_opinion_cluster = Column(
        Integer,
        index=True,
        nullable=True,
        comment="The cluster_id of the resolved citation for case law",
    )
    resolved_text = Column(String, nullable=True, comment="The resolved citation text")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationship
    opinion_cluster = relationship(
        "OpinionClusterExtraction", back_populates="citations"
    )
