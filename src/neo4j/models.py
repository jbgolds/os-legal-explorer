from neomodel import (
    StructuredNode,
    StructuredRel,
    StringProperty,
    IntegerProperty,
    DateTimeProperty,
    DateProperty,
    RelationshipTo,
    RelationshipFrom,
    BooleanProperty,
    ArrayProperty,
    UniqueIdProperty,
    JSONProperty,
    ZeroOrMore,
    db,
)
from datetime import datetime
from src.llm_extraction.models import CitationType, CitationTreatment, OpinionType
import asyncio


class CitesRel(StructuredRel):
    """
    Represents a citation relationship between opinions with enriched metadata.
    All fields except source are optional to accommodate different data sources.
    """

    # Required fields
    data_source = StringProperty(required=True)  # Source dataset identifier
    version = IntegerProperty(default=1)

    # Metadata fields - index only treatment as it's commonly queried
    treatment = StringProperty(
        choices=CitationTreatment._member_map_, index=True, default=None
    )  # Frequently filtered
    relevance = IntegerProperty(default=None)
    reasoning = StringProperty(default=None)
    citation_text = StringProperty(default=None)
    page_number = IntegerProperty(default=None)

    # Temporal fields
    timestamp = DateTimeProperty(default=lambda: datetime.now())
    opinion_section = StringProperty(default=None)  # majority, dissent, or concurrent

    # Audit trail
    other_metadata_versions = JSONProperty(default=list)

    # Flag for default data created during error handling
    is_default_data = BooleanProperty(default=False, index=True)


class BaseNode(StructuredNode):
    """Base class for all nodes with common properties and methods."""

    # Common timestamps for all nodes
    created_at = DateTimeProperty(default=lambda: datetime.now())
    updated_at = DateTimeProperty(default=lambda: datetime.now())

    # Abstract base class
    __abstract_node__ = True

    def pre_save(self):
        """Update timestamp before saving"""
        self.updated_at = datetime.now()


class Citation(BaseNode):
    """
    Base citation node for all types of citations.
    The type field determines how the citation should be interpreted.
    Additional metadata can be stored in the metadata JSON field.
    """

    # Core fields with indexes
    citation_text = StringProperty(unique_index=True, required=True)
    type = StringProperty(required=True, choices=CitationType._member_map_, index=True)
    metadata = JSONProperty(default=dict)

    # Relationships
    cited_by = RelationshipFrom(
        "Citation", "CITES", model=CitesRel, cardinality=ZeroOrMore
    )
    cites = RelationshipTo("Citation", "CITES", model=CitesRel, cardinality=ZeroOrMore)

    @property
    def citation_count(self):
        """Get real-time citation count from relationships"""
        return len(self.cited_by.all())

    @classmethod
    def get_citations_by_type(cls, citation_type: CitationType):
        """Get all citations of a specific type"""
        return cls.nodes.filter(type=citation_type)


class Opinion(Citation):
    """
    Specialized Citation node representing judicial opinions.
    Inherits base citation functionality and adds opinion-specific fields.

    IMPORTANT: The full text of opinions should NOT be stored in Neo4j.
    Original opinion text is stored in the PostgreSQL database in the opinion_text table.
    Neo4j should only contain metadata and citation relationships for efficient graph traversal.
    """

    # Set the type automatically for all Opinion instances
    type = StringProperty(default=CitationType.judicial_opinion.value)

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        if self.type != CitationType.judicial_opinion:
            raise ValueError("Opinions must have type CitationType.judicial_opinion")

    # Primary identifier
    cluster_id = IntegerProperty(unique_index=True, required=True)

    # Core metadata - index frequently queried fields
    date_filed = DateProperty(index=True)
    case_name = StringProperty()
    docket_id = IntegerProperty(default=None)
    docket_number = StringProperty(default=None)
    court_id = IntegerProperty(index=True)
    court_name = StringProperty(default=None)

    # Opinion type and voting data
    opinion_type = StringProperty(choices=OpinionType._member_map_, default=None)
    scdb_votes_majority = IntegerProperty(default=None)
    scdb_votes_minority = IntegerProperty(default=None)

    # Original database IDs
    opinion_id = IntegerProperty(default=None)
    docket_db_id = IntegerProperty(default=None)
    court_db_id = IntegerProperty(default=None)

    # Enhanced metadata
    ai_summary = StringProperty(default=None)

    # Flag for default data created during error handling
    is_default_data = BooleanProperty(default=False, index=True)

    # Strategic composite indexes for common query patterns
    __indexes__ = {
        # Most common query pattern: finding opinions by court within a date range
        "court_date_idx": {"fields": ["court_id", "date_filed"], "type": "composite"},
    }
