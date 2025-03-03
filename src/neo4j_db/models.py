from neomodel import (
    StructuredNode,
    StructuredRel,
    StringProperty,
    IntegerProperty,
    DateTimeProperty,
    DateProperty,
    RelationshipTo,
    JSONProperty,
    ZeroOrMore,
    db,
    DateTimeFormatProperty,
)
from neomodel.properties import validator
from datetime import datetime, date
from src.llm_extraction.models import CitationType, CitationTreatment, OpinionSection
import asyncio
from typing import Type
import json


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


class CustomJSONProperty(JSONProperty):
    """Custom JSONProperty that can handle datetime objects."""

    @validator
    def deflate(self, value, obj=None):
        """Convert value to JSON string with datetime support."""
        if value is None:
            return None
        return json.dumps(value, cls=DateTimeEncoder, ensure_ascii=self.ensure_ascii)


class CitesRel(StructuredRel):
    """
    Represents a citation relationship between legal documents with enriched metadata.
    All fields except data_source are optional to accommodate different data sources.
    The citation_text is stored on the relationship, not on the nodes.
    """

    # Required fields
    data_source = StringProperty(required=True)  # Source dataset identifier
    version = IntegerProperty(default=1)

    # Citation text - how one document refers to another (properly belongs on the relationship)
    citation_text = StringProperty(default=None)

    # Metadata fields - index only treatment as it's commonly queried
    treatment = StringProperty(
        choices=CitationTreatment._value2member_map_, index=True, default=None
    )  # Frequently filtered
    relevance = IntegerProperty(default=None)
    reasoning = StringProperty(default=None)
    page_number = IntegerProperty(default=None)

    opinion_section = StringProperty(
        choices=OpinionSection._value2member_map_, default=None
    )
    # Temporal fields
    timestamp = DateTimeFormatProperty(format="%Y-%m-%d %H:%M:%S", default_now=True)

    # Audit trail - using CustomJSONProperty instead of JSONProperty
    other_metadata_versions = CustomJSONProperty(default=list)


class LegalDocument(StructuredNode):
    """
    Generic class for all legal document nodes with common properties and methods.
    This replaces the previous multiple node classes with a single, more flexible class.
    """

    # Common timestamps for all nodes
    created_at = DateTimeProperty(default=lambda: datetime.now())
    updated_at = DateTimeProperty(default=lambda: datetime.now())

    # Common relationships for all document types
    cites = RelationshipTo(
        cls_name="LegalDocument",
        relation_type="CITES",
        model=CitesRel,
        cardinality=ZeroOrMore,
    )

    # Generic primary identifier - used to relate to records in other databases
    primary_id = StringProperty(index=True)  # Can store any ID as string
    primary_table = StringProperty(
        required=True, index=True
    )  # Stores source table name

    title = StringProperty(
        default=None
    )  # Generic title field (case_name for opinions, etc.)

    # Additional metadata
    metadata = JSONProperty(default=dict)  # Stores type-specific metadata

    # Original citation text or identifier string
    citation_string = StringProperty(
        default=None, required=True
    )  # Stores the original citation string

    # Strategic composite indexes for common query patterns
    __indexes__ = {
        # Existing indexes
        "court_date": ("court_id", "date_filed"),
        "primary_key": ("type", "primary_id"),
        # New unique constraint
        "citation_unique": {"fields": ("type", "citation_string"), "unique": True},
    }

    def pre_save(self):
        """Update timestamp before saving"""
        self.updated_at = datetime.now()

    @classmethod
    def get_or_create(cls, citation_string, **kwargs):
        """Get or create a document, always ensuring citation_string is set"""
        # Convert numeric timestamps to datetime objects
        if "updated_at" in kwargs and isinstance(kwargs["updated_at"], (int, float)):
            kwargs["updated_at"] = datetime.fromtimestamp(kwargs["updated_at"])

        if "created_at" in kwargs and isinstance(kwargs["created_at"], (int, float)):
            kwargs["created_at"] = datetime.fromtimestamp(kwargs["created_at"])

        # Initialize doc to None
        doc = None

        # first try and find via primary_id, then check citation_string
        if "primary_id" in kwargs:
            doc = cls.nodes.first_or_none(primary_id=kwargs["primary_id"])

        if not doc:
            doc = cls.nodes.first_or_none(citation_string=citation_string)

        if not doc:
            # Create a new document
            kwargs["citation_string"] = citation_string
            doc = cls(**kwargs)
            doc.save()
        return doc


class Opinion(LegalDocument):
    """
    Node representing judicial opinions.

    IMPORTANT: The full text of opinions should NOT be stored in Neo4j.
    Original opinion text is stored in the PostgreSQL database in the opinion_text table.
    Neo4j should only contain metadata and citation relationships for efficient graph traversal.
    """

    # Opinion-specific fields
    ai_summary = StringProperty(default=None)
    case_name = StringProperty(default=None)
    docket_id = IntegerProperty(default=None)
    docket_number = StringProperty(default=None)
    court_id = IntegerProperty(index=True)
    court_name = StringProperty(default=None)

    date_filed = DateProperty(index=True)

    # Original database IDs
    opinion_id = IntegerProperty(default=None)
    docket_db_id = IntegerProperty(default=None)
    court_db_id = IntegerProperty(default=None)

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "opinion_cluster"

    @classmethod
    def get_or_create_from_cluster_id(cls, cluster_id, citation_string, **kwargs):
        """Get an Opinion by cluster_id or create it if it doesn't exist

        Args:
            cluster_id: The CourtListener cluster ID
            **kwargs: Additional properties to set when creating the opinion

        Returns:
            The Opinion node (either existing or newly created)
        """
        # Ensure datetime conversion for any timestamp inputs
        if "updated_at" in kwargs and isinstance(kwargs["updated_at"], (int, float)):
            kwargs["updated_at"] = datetime.fromtimestamp(kwargs["updated_at"])

        if "created_at" in kwargs and isinstance(kwargs["created_at"], (int, float)):
            kwargs["created_at"] = datetime.fromtimestamp(kwargs["created_at"])

        # Convert cluster_id to string for primary_id
        if not cluster_id and not citation_string:
            raise ValueError("Cluster ID or citation string is required")

        primary_id = str(cluster_id) if cluster_id else None
        if primary_id:
            opinion = cls.nodes.first_or_none(primary_id=primary_id)
        elif citation_string:
            opinion = cls.nodes.first_or_none(citation_string=citation_string)

        # Create if it doesn't exist
        if not opinion:
            # Set required fields
            kwargs.update(
                {
                    "primary_id": primary_id,
                    "primary_table": "opinion_cluster",
                }
            )
            if citation_string:
                kwargs["citation_string"] = citation_string

            # Create new opinion
            opinion = cls(**kwargs)
            opinion.save()

        return opinion
        # if len(opinion) > 1:
        #     raise ValueError(
        #         f"Multiple opinions found for cluster_id {cluster_id} and citation_string {citation_string}"
        #     )
        # elif len(opinion) == 1:
        #     return opinion[0]
        # else:
        #     raise ValueError(
        #         f"No opinion found for cluster_id {cluster_id} and citation_string {citation_string}"
        #     )


class StatutesCodesRegulation(LegalDocument):
    """
    Node representing statutes, codes, and regulations.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "statutes"


class CongressionalReport(LegalDocument):
    """
    Node representing congressional reports.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "congressional_reports"


class LawReview(LegalDocument):
    """
    Node representing law reviews.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "law_reviews"


class LegalDictionary(LegalDocument):
    """
    Node representing legal dictionaries.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "legal_dictionaries"


class OtherLegalDocument(LegalDocument):
    """
    Node representing other legal documents.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "other_legal_documents"


class ConstitutionalDocument(LegalDocument):
    """
    Node representing constitutional documents.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "constitutional_documents"


class AdministrativeAgencyRuling(LegalDocument):
    """
    Node representing administrative agency rulings.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "admin_rulings"


class ExternalSubmission(LegalDocument):
    """
    Node representing external submissions.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "submissions"


class ElectronicResource(LegalDocument):
    """
    Node representing electronic resources.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "electronic_resources"


# Mapping to help with backwards compatibility and migration
CITATION_TYPE_TO_NODE_TYPE: dict[CitationType, Type[LegalDocument]] = {
    CitationType.judicial_opinion: Opinion,
    CitationType.statutes_codes_regulations: StatutesCodesRegulation,
    CitationType.constitution: ConstitutionalDocument,
    CitationType.administrative_agency_ruling: AdministrativeAgencyRuling,
    CitationType.congressional_report: CongressionalReport,
    CitationType.external_submission: ExternalSubmission,
    CitationType.electronic_resource: ElectronicResource,
    CitationType.law_review: LawReview,
    CitationType.legal_dictionary: LegalDictionary,
    CitationType.other: OtherLegalDocument,
}
