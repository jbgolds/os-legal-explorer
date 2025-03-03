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

    # Temporal fields
    timestamp = DateTimeProperty(default=lambda: datetime.now())
    opinion_section = StringProperty(default=None)  # majority, dissent, or concurrent

    # Audit trail
    other_metadata_versions = JSONProperty(default=list)

    # Flag for default data created during error handling
    is_default_data = BooleanProperty(default=False, index=True)


class LegalDocument(StructuredNode):
    """
    Generic class for all legal document nodes with common properties and methods.
    This replaces the previous multiple node classes with a single, more flexible class.
    """

    # Common timestamps for all nodes
    created_at = DateTimeProperty(default=lambda: datetime.now())
    updated_at = DateTimeProperty(default=lambda: datetime.now())

    # Common relationships for all document types
    cited_by = RelationshipFrom(
        "LegalDocument", "CITES", model=CitesRel, cardinality=ZeroOrMore
    )
    cites = RelationshipTo(
        "LegalDocument", "CITES", model=CitesRel, cardinality=ZeroOrMore
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
        # Try to find existing document

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

    # Opinion type and voting data
    opinion_type = StringProperty(choices=OpinionType._value2member_map_, default=None)
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
        # Convert cluster_id to string for primary_id
        primary_id = str(cluster_id)

        # Try to find existing opinion
        # first try and find via primary_id, then check citation_string

        opinion = cls.nodes.first_or_none(
            primary_id=primary_id, type=CitationType.judicial_opinion.value
        )
        if not opinion:
            opinion = cls.nodes.first_or_none(
                citation_string=citation_string,
                type=CitationType.judicial_opinion.value,
            )

        # Create if it doesn't exist
        if not opinion:
            # Set required fields
            kwargs.update(
                {
                    "primary_id": primary_id,
                    "primary_table": "opinion_cluster",
                    "type": CitationType.judicial_opinion.value,
                    # If citation_string not provided, use cluster_id as fallback
                    "citation_string": citation_string,
                }
            )

            # Create new opinion
            opinion = cls(**kwargs)
            opinion.save()

        return opinion


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
CITATION_TYPE_TO_NODE_TYPE = {
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
