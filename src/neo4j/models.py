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


class BaseNode(StructuredNode):
    """Base class for all nodes with common properties and methods."""

    # Common timestamps for all nodes
    created_at = DateTimeProperty(default=lambda: datetime.now())
    updated_at = DateTimeProperty(default=lambda: datetime.now())

    # Common relationships for all document types
    cited_by = RelationshipFrom(
        "BaseNode", "CITES", model=CitesRel, cardinality=ZeroOrMore
    )
    cites = RelationshipTo("BaseNode", "CITES", model=CitesRel, cardinality=ZeroOrMore)

    # Flag for default data created during error handling
    is_default_data = BooleanProperty(default=False, index=True)

    # Additional metadata
    metadata = JSONProperty(default=dict)

    # Type is a common field across all legal documents
    type = StringProperty(
        required=True, choices=CitationType._value2member_map_, index=True
    )

    # Abstract base class
    __abstract_node__ = True

    def pre_save(self):
        """Update timestamp before saving"""
        self.updated_at = datetime.now()

    @property
    def citation_count(self):
        """Get real-time citation count from relationships"""
        return len(self.cited_by.all())


class Opinion(BaseNode):
    """
    Node representing judicial opinions.

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
    opinion_type = StringProperty(choices=OpinionType._value2member_map_, default=None)
    scdb_votes_majority = IntegerProperty(default=None)
    scdb_votes_minority = IntegerProperty(default=None)

    # Original database IDs
    opinion_id = IntegerProperty(default=None)
    docket_db_id = IntegerProperty(default=None)
    court_db_id = IntegerProperty(default=None)

    # Enhanced metadata
    ai_summary = StringProperty(default=None)

    # Strategic composite indexes for common query patterns
    __indexes__ = {
        # Most common query pattern: finding opinions by court within a date range
        "court_date_idx": {"fields": ["court_id", "date_filed"], "type": "composite"},
    }


class Statute(BaseNode):
    """Node representing statutes, codes, and regulations."""

    type = StringProperty(default=CitationType.statutes_codes_regulations.value)

    # Identifiers
    code_string = StringProperty(required=True)

    def pre_save(self):
        """Ensure type is always correct"""
        super().pre_save()
        if self.type != CitationType.statutes_codes_regulations:
            raise ValueError(
                "Statutes must have type CitationType.statutes_codes_regulations"
            )


class Constitution(BaseNode):
    """Node representing constitutional provisions."""

    type = StringProperty(default=CitationType.constitution.value)

    # Identifiers
    constitution_section_string = StringProperty(equired=True)

    def pre_save(self):
        """Ensure type is always correct"""
        super().pre_save()
        if self.type != CitationType.constitution:
            raise ValueError("Constitution must have type CitationType.constitution")


class AdministrativeRuling(BaseNode):
    """Node representing administrative agency rulings."""

    type = StringProperty(default=CitationType.administrative_agency_ruling.value)

    # Identifiers
    ruling_string = StringProperty(required=True)

    def pre_save(self):
        """Ensure type is always correct"""
        super().pre_save()
        if self.type != CitationType.administrative_agency_ruling:
            raise ValueError(
                "Administrative rulings must have type CitationType.administrative_agency_ruling"
            )


class CongressionalReport(BaseNode):
    """Node representing congressional reports."""

    type = StringProperty(default=CitationType.congressional_report.value)

    # Identifiers
    report_string = StringProperty(required=True)

    def pre_save(self):
        """Ensure type is always correct"""
        super().pre_save()
        if self.type != CitationType.congressional_report:
            raise ValueError(
                "Congressional reports must have type CitationType.congressional_report"
            )


class ExternalSubmission(BaseNode):
    """Node representing external submissions (briefs, memoranda, etc.)."""

    type = StringProperty(default=CitationType.external_submission.value)

    # Identifiers
    submission_string = StringProperty(required=True)

    def pre_save(self):
        """Ensure type is always correct"""
        super().pre_save()
        if self.type != CitationType.external_submission:
            raise ValueError(
                "External submissions must have type CitationType.external_submission"
            )


class ElectronicResource(BaseNode):
    """Node representing electronic resources."""

    type = StringProperty(default=CitationType.electronic_resource.value)

    # Identifiers
    resource_string = StringProperty(required=True)

    def pre_save(self):
        """Ensure type is always correct"""
        super().pre_save()
        if self.type != CitationType.electronic_resource:
            raise ValueError(
                "Electronic resources must have type CitationType.electronic_resource"
            )


class LawReview(BaseNode):
    """Node representing law review articles."""

    type = StringProperty(default=CitationType.law_review.value)

    # Identifiers
    article_string = StringProperty(required=True)

    def pre_save(self):
        """Ensure type is always correct"""
        super().pre_save()
        if self.type != CitationType.law_review:
            raise ValueError("Law reviews must have type CitationType.law_review")


class LegalDictionary(BaseNode):
    """Node representing legal dictionary entries."""

    type = StringProperty(default=CitationType.legal_dictionary.value)

    # Identifiers
    entry_id = StringProperty(required=True)

    def pre_save(self):
        """Ensure type is always correct"""
        super().pre_save()
        if self.type != CitationType.legal_dictionary:
            raise ValueError(
                "Legal dictionary entries must have type CitationType.legal_dictionary"
            )


class OtherLegalDocument(BaseNode):
    """Node representing other legal documents not fitting the above categories."""

    type = StringProperty(default=CitationType.other.value)

    # Identifiers
    document_string = StringProperty(required=True)

    def pre_save(self):
        """Ensure type is always correct"""
        super().pre_save()
        if self.type != CitationType.other:
            raise ValueError("Other legal documents must have type CitationType.other")
