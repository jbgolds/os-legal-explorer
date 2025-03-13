"""
Bugs:
- The CitesRel Relationship has a timestamp field, but also a created_at + updated_at field on the LegalDocument class, but I don't see it on neo4j.



"""
import logging
import json
from datetime import date, datetime
from typing import Optional, Type

from neomodel import (DateProperty, DateTimeFormatProperty, DateTimeProperty,
                      IntegerProperty, JSONProperty, RelationshipTo,
                      StringProperty, AsyncStructuredNode, AsyncStructuredRel, AsyncRelationshipTo, AsyncZeroOrMore
                      , adb)
from neomodel.properties import validator

from src.llm_extraction.models import (CitationTreatment, CitationType,
                                       OpinionSection)

logger = logging.getLogger(__name__)

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


class CitesRel(AsyncStructuredRel):
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

    async def update_history(self, new_props: dict):
        """Update the history of the relationship with a new version."""
        current_props = self.__properties__.copy()

        # Remove other_metadata_versions from the properties to avoid circular references
        if "other_metadata_versions" in current_props:
            current_props.pop("other_metadata_versions")

        # we deal with version ourselves, not set by client.
        if "version" in new_props:
            new_props.pop("version")

        # Append the current properties to other_metadata_versions
        logger.debug("Current other_metadata_versions: %s", self.other_metadata_versions)
        cur_version = self.other_metadata_versions or []
        # Get the end node element_id synchronously
        en = await self.end_node()
        current_props["cites"] = en.element_id
        # Convert to list if it's not already
        if not isinstance(cur_version, list):
            cur_version = [cur_version] if cur_version else []
            
        cur_version.append(current_props)

        self.other_metadata_versions = cur_version
        if self.version is None:
            self.version = 1
        elif isinstance(self.version, int):
            self.version = self.version + 1
        else:
            raise ValueError(f"Version must be an integer, got {type(self.version)}")
        
        for key, value in new_props.items():
            if key in self.defined_properties().keys():
                setattr(self, key, value)

        await self.save()




class LegalDocument(AsyncStructuredNode):
    """
    Generic class for all legal document nodes with common properties and methods.
    This replaces the previous multiple node classes with a single, more flexible class.
    """
    # TODO, think should be have been set to abstract_node, but do not want to break anything.
    # __abstract_node__ = True
    # Common timestamps for all nodes
    created_at = DateTimeProperty(default=lambda: datetime.now())
    updated_at = DateTimeProperty(default=lambda: datetime.now())

    # Common relationships for all document types
    cites = AsyncRelationshipTo(
        cls_name="LegalDocument",
        relation_type="CITES",
        model=CitesRel,
        cardinality=AsyncZeroOrMore,
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
    async def get_or_create_from_cluster_id(cls, cluster_id, citation_string, **kwargs):
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
            opinion: Optional[Opinion] = await cls.nodes.first_or_none(primary_id=primary_id)
        elif citation_string:
            opinion: Optional[Opinion] = await cls.nodes.first_or_none(citation_string=citation_string)
        

        if not opinion:
            opinion = cls()
            # Set primary_id if available
            if primary_id:
                opinion.primary_id = primary_id
            # Ensure citation_string is set
            opinion.citation_string = citation_string
    
        if not opinion.ai_summary and kwargs.get("ai_summary"):
            opinion.ai_summary = kwargs["ai_summary"]
        if not opinion.date_filed and kwargs.get("date_filed"):
            opinion.date_filed = kwargs["date_filed"]

        # Check if citation_string needs to be updated
        if opinion.citation_string and "cluster-" in str(opinion.citation_string) and citation_string:
            opinion.citation_string = citation_string
            
        # Set required fields
        for key, value in kwargs.items():
            if key in cls.defined_properties().keys():
                setattr(opinion, key, value)

        # Save the opinion
        await opinion.save()

        return opinion
        


class StatutesCodesRegulation(LegalDocument):
    """
    Node representing statutes, codes, and regulations.
    """

    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        # TODO change this to mapping, instead of hardcoding
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
    primary_table = "submissions"
    def pre_save(self):
        """Ensure opinion type is always correct"""
        super().pre_save()
        # Ensure primary_table is correctly set
        self.primary_table = "submissions"


class ElectronicResource(LegalDocument):
    """
    Node representing electronic resources.
    """ 
    primary_table = "electronic_resources"
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
