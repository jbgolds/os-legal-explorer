from argparse import Action
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import StrEnum
from typing import List, Optional
import json
from json_repair import repair_json
from src.postgres.db_utils import find_cluster_id
from datetime import datetime

from sql_models.models import CitationExtraction, OpinionClusterExtraction

# https://www.law.cornell.edu/citation/6-300
# https://support.vlex.com/how-to/navigate-documents/case-law/treatment-types#undefined
# https://www.lexisnexis.com/pdf/lexis-advance/Shepards-Signal-Indicators-and-analysis-phrases-Final.pdf?srsltid=AfmBOopBSKqOSAAcyYrvQrjavGWoE7ERkWbUrqwM8BrFkYH1RaedBziZ
# https://www.paxton.ai/post/introducing-the-paxton-ai-citator-setting-new-benchmarks-in-legal-research
# from above ^ https://scholarship.law.wm.edu/cgi/viewcontent.cgi?article=1130&context=libpubs; Evaluating Shepard's, KeyCite, and Bcite for Case Validation Accuracy. INACCURATE!

# NOTE to self: Need to think about this model/data in that it's point in time reasoning, Not reasoning today. Need to later combine.


signals = """
(a) Signals that indicate support:
- E.g., : Authority states the proposition with which the citation is associated. Other authorities, not cited, do as well «e.g.». "E.g." used with other signals (in which case it is preceded by a comma) similarly indicates the existence of other authorities not cited.
- Accord: Used following citation to authority referred to in text when there are additional authorities that either state or clearly support the proposition with which the citation is associated, but the text quotes only one. Similarly, the law of one jurisdiction may be cited as being in accord with that of another «e.g.».
- See: Authority supports the proposition with which the citation is associated either implicitly or in the form of dicta «e.g.» .
- See also: Authority is additional support for the proposition with which the citation is associated (but less direct than that indicated by "see" or "accord"). "See also" is commonly used to refer readers to authorities already cited or discussed «e.g.». Generally, it is helpful to include a parenthetical explanation of the source material's relevance following a citation introduced by "see also."
- Cf. : Authority supports by analogy. "Cf." literally means "compare." The citation will only appear relevant to the reader if it is explained. Consequently, in most cases a parenthetical explanations of the analogy should be included «e.g.».
- Followed by: Authority supports the proposition with which the citation is associated.

(b) Signals that suggest a useful comparison.
- Compare ... with ... : Comparison of authorities that supports proposition. Either side of the comparison can have more than one item linked with "and" «e.g.». Generally, a parenthetical explanation of the comparison should be furnished.

(c) Signals that indicate contradiction.
- Contra: Authority directly states the contrary of the proposition with which the citation is associated «e.g.».
- But see: Authority clearly supports the contrary of the proposition with which citation is associated «e.g.».
- But cf.: Authority supports the contrary of the position with which the citation is associated by analogy. In most cases a parenthetical explanations of the analogy should be furnished. The word "but" is omitted from the signal when it follows another negative signal «e.g.».
- Overruled by: The citing case expressly overrules or disapproves all or part of the case cited.
- Abrogated by: The citing case effectively, but not explicitly, overrules or departs from the case cited.
- Superseded by: The citing reference—typically a session law, other record of legislative action or a record of administrative action— supersedes the statute, regulation or order cited.

(d) Signals that indicate background material.
- See generally: Authority presents useful background. Parenthetical explanations of the source materials' relevance are generally useful «e.g.».

(e) Combining a signal with "e.g."
- E.g.,: In addition to the cited authority, there are numerous others that state, support, or contradict the proposition (with the other signal indicating which) but citation to them would not be helpful or necessary. The preceding signal is separated from "e.g." by a comma «e.g.».
"""


class OpinionSection(StrEnum):
    majority = "MAJORITY"
    concurring = "CONCURRING"
    dissenting = "DISSENTING"


class CitationType(StrEnum):
    judicial_opinion = "judicial_opinion"
    statutes_codes_regulations = "statutes_codes_regulations"
    constitution = "constitution"
    administrative_agency_ruling = "administrative_agency_ruling"
    congressional_report = "congressional_report"
    external_submission = "external_submission"
    electronic_resource = "electronic_resource"
    law_review = "law_review"
    legal_dictionary = "legal_dictionary"
    other = "other"


class CitationTreatment(StrEnum):
    positive = "POSITIVE"  # Only when explicitly relied on as key basis
    negative = "NEGATIVE"  # Explicitly disagrees with, distinguishes, or limits
    caution = "CAUTION"  # Expresses doubts or declines to extend
    neutral = "NEUTRAL"  # Default for background/general reference


class Citation(BaseModel):
    """Model for a single citation extracted from a court opinion."""
    page_number: int = Field(
        ...,
        description="The page number where this citation appears (must be a positive integer).",
    )
    citation_text: str = Field(
        ...,
        description="The entirety of the extracted citation text value, resolved back to the original citation when possible.",
    )
    reasoning: str = Field(
        ...,
        description="Detailed analysis explaining this citation's context and relevance (2-4 sentences).",
    )
    type: CitationType = Field(
        ...,
        description="The citation type (e.g., 'judicial_opinion').",
    )
    treatment: CitationTreatment = Field(
        ...,
        description="The citation treatment (POSITIVE, NEGATIVE, CAUTION, or NEUTRAL).",
    )
    relevance: int = Field(
        ...,
        description="Relevance score between 1 (lowest) and 4 (highest).",
    )

    class Config:
        use_enum_values = True

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v):
        if isinstance(v, str):
            # Map common input values to enum values
            mapping = {
                "Judicial Opinion": CitationType.judicial_opinion,
                "Statute/Code/Regulation/Rule": CitationType.statutes_codes_regulations,
                "Statute/Code/Regulation": CitationType.statutes_codes_regulations,
                "Constitution": CitationType.constitution,
                "Administrative/Agency Ruling": CitationType.administrative_agency_ruling,
                "Congressional Report": CitationType.congressional_report,
                "External Submission": CitationType.external_submission,
                "Electronic Resource": CitationType.electronic_resource,
                "Law Review": CitationType.law_review,
                "Legal Dictionary": CitationType.legal_dictionary,
                "Other": CitationType.other,
            }

            # Try direct mapping first
            if v in mapping:
                return mapping[v]
            # Try case-insensitive match
            v_lower = v.lower()
            for input_val, enum_val in mapping.items():
                if v_lower == input_val.lower():
                    return enum_val
            # If no match found, try direct enum conversion
            try:
                return CitationType(v)
            except ValueError:
                print(f"Invalid citation type string, defaulting to 'other': {v}")
                return CitationType.other
        elif isinstance(v, CitationType):
            return v
        else:
            print(f"Invalid citation type, defaulting to 'other': {v}")
            return CitationType.other


class CitationAnalysis(BaseModel):
    """Model for a single court opinions complete citation analysis, including the date, brief summary, and citations."""
    date: str = Field(
        ...,
        description="The date of the day the opinion was published, in format YYYY-MM-DD",
    )
    brief_summary: str = Field(
        ..., description="3-5 sentences describing the core holding"
    )
    majority_opinion_citations: List[Citation] = Field(
        ...,
        description="List of citations from the majority opinion, including footnotes.",
    )
    concurrent_opinion_citations: List[Citation] = Field(
        ...,
        description="List of citations from concurring opinions, including footnotes.",
    )
    dissenting_citations: List[Citation] = Field(
        ...,
        description="List of citations from dissenting opinions, including footnotes.",
    )

    model_config = ConfigDict(
        json_encoders={str: lambda v: v}  # Preserve Unicode characters in JSON
    )

    # function to count length of all strings in this model
    def count_length(self) -> int:
        sum = 0
        for field in self.model_fields:
            if field.annotation == list or field.annotation == List:
                for item in getattr(self, field.name):
                    if isinstance(item, Citation):
                        sum += item.count_citation_length()
                    # for strings for `notes`
                    elif isinstance(item, str):
                        sum += len(item)
            else:
                sum += len(getattr(self, field.name))
        return sum


class CitationResolved(Citation):
    """This is for the resolved citation, which includes the cluster_id of the resolved citation for case law;"""
    resolved_opinion_cluster: int | None = Field(
        ...,
        description="The cluster_id of the resolved citation for case law; not for statutes or regulations.",
    )
    resolved_statute_code_regulation_rule: int | None = Field(
        ...,
        description="TBD, NEED TO CREATE THIS DATABASE TABLE",
    )


def resolve_citation(citation: Citation) -> CitationResolved:
    """Convert a Citation to a CitationResolved with a postgres db lookup."""
    if citation.type == CitationType.judicial_opinion:
        return CitationResolved(
            **citation.model_dump(),  # Copy all existing fields
            resolved_opinion_cluster=find_cluster_id(citation.citation_text),
            resolved_statute_code_regulation_rule=0,
        )
    else:
        return CitationResolved(
            **citation.model_dump(),
            resolved_opinion_cluster=None,
            resolved_statute_code_regulation_rule=None,
        )


# TODO Overlap with CitationAnalysis.
class CombinedResolvedCitationAnalysis(BaseModel):
    date: str
    cluster_id: int
    brief_summary: str
    majority_opinion_citations: list[CitationResolved]
    concurrent_opinion_citations: list[CitationResolved]
    dissenting_citations: list[CitationResolved]

    @classmethod
    def from_citations_json(cls, citations: str, cluster_id: int):
        """Expects a JSON string representing the CitationAnalysis model, from a
        request to Gemini, so checks the `parsed` key."""
        citation_data = None
        if not cluster_id:
            raise ValueError("Cluster ID is required")

        # Handle string input
        if isinstance(citations, str):
            citations = json.loads(citations)

        # If citations is a dict with 'parsed' key, use that
        if isinstance(citations, dict):

            if "parsed" in citations and citations["parsed"] is not None:
                citation_data = citations["parsed"]

                # if something happened with response, and json malformed, try to parse the text
            if isinstance(citation_data, str):
                citation_data = json.loads(citation_data)
        else:
            citation_data = citations

        if citation_data is None:
            # try and parse the text
            try:
                citation_data = citations["candidates"][0]["content"]["parts"][0][
                    "text"
                ]

                citation_data = repair_json(citation_data)
                # step 3, load into json
                citation_data = json.loads(citation_data)

            except Exception as e:
                print("Could not parse citation for cluster_id: %s, %s", cluster_id, e)
                citation_data = None

        # print(type(citation_data))
        # print(citation_data)
        # Convert to CitationAnalysis model and use existing from_citations method
        majority_opinion_citations = []
        concurrent_opinion_citations = []
        dissenting_citations = []
        keys = [
            "majority_opinion_citations",
            "concurrent_opinion_citations",
            "dissenting_citations",
        ]
        # if not all keys are present, only lookup found keys
        if not all(key in citation_data.keys() for key in keys):
            keys = [key for key in keys if key in citation_data.keys()]

        for key in keys:
            for citation in citation_data[key]:
                # compare dictionary keys of citation with Citation model
                if not all(
                    field in citation.keys() for field in Citation.model_fields.keys()
                ):
                    continue
                else:
                    Citation(
                        citation_text=citation["citation_text"],
                        reasoning=citation["reasoning"],
                        type=citation["type"],
                        treatment=citation["treatment"],
                        relevance=citation["relevance"],
                    )
                    if key == "majority_opinion_citations":
                        majority_opinion_citations.append(citation)
                    elif key == "concurrent_opinion_citations":
                        concurrent_opinion_citations.append(citation)
                    elif key == "dissenting_citations":
                        dissenting_citations.append(citation)
        citation_analysis = CitationAnalysis(
            date=citation_data["date"],
            brief_summary=citation_data["brief_summary"],
            majority_opinion_citations=majority_opinion_citations,
            concurrent_opinion_citations=concurrent_opinion_citations,
            dissenting_citations=dissenting_citations,
        )

        # citation_analysis = CitationAnalysis.model_validate(citation_data)
        return cls.from_citations([citation_analysis], cluster_id)

    @classmethod
    def from_citations(
        cls,
        citations: list[CitationAnalysis],
        cluster_id: int,
    ) -> "CombinedResolvedCitationAnalysis":

        return cls(
            date=citations[0].date,
            cluster_id=cluster_id,
            brief_summary=citations[0].brief_summary,
            majority_opinion_citations=[
                resolve_citation(citation)
                for citation_analysis in citations
                for citation in citation_analysis.majority_opinion_citations
            ],
            concurrent_opinion_citations=[
                resolve_citation(citation)
                for citation_analysis in citations
                for citation in citation_analysis.concurrent_opinion_citations
            ],
            dissenting_citations=[
                resolve_citation(citation)
                for citation_analysis in citations
                for citation in citation_analysis.dissenting_citations
            ],
        )

    def get_opinion_nodes(self) -> dict:
        """Return a dictionary of unique opinion nodes needed for this citation analysis.
        The key is the opinion cluster_id and the value is a dict of properties.
        For the citing opinion, includes 'date_filed'; for others, only 'cluster_id'."""
        nodes = {}
        # Add the citing opinion node
        nodes[self.cluster_id] = {
            "cluster_id": self.cluster_id,
            "date_filed": self.date,
            "type": "judicial_opinion",
        }

        # Helper function to process a list of citations
        def process_citations(citations):
            for citation in citations:
                if citation.resolved_opinion_cluster is not None:
                    if citation.resolved_opinion_cluster not in nodes:
                        nodes[citation.resolved_opinion_cluster] = {
                            "cluster_id": citation.resolved_opinion_cluster,
                            "type": citation.type,  # Add citation type from the model
                        }

        process_citations(self.majority_opinion_citations)
        process_citations(self.concurrent_opinion_citations)
        process_citations(self.dissenting_citations)

        return nodes


def to_sql_models(
    combined: CombinedResolvedCitationAnalysis,
) -> tuple[OpinionClusterExtraction, list[CitationExtraction]]:
    """Convert a CombinedResolvedCitationAnalysis to SQL models.

    Returns:
        Tuple containing:
        - OpinionClusterExtraction instance
        - List of CitationExtraction instances
    """
    # Create the opinion cluster extraction
    cluster = OpinionClusterExtraction(
        cluster_id=combined.cluster_id,
        date_filed=datetime.strptime(combined.date, "%Y-%m-%d"),
        brief_summary=combined.brief_summary,
    )

    citations = []

    # Helper to process citations from a section
    def process_section_citations(citation_list: list[CitationResolved]):
        for citation in citation_list:
            cite = CitationExtraction(
                opinion_cluster_extraction_id=cluster.id,  # This will be set after DB insert
                section=Action,
                citation_type=citation.type,
                citation_text=citation.citation_text,
                page_number=citation.page_number,
                treatment=citation.treatment,
                relevance=citation.relevance,
                reasoning=citation.reasoning,
                resolved_opinion_cluster=citation.resolved_opinion_cluster,
                resolved_text=citation.citation_text,  # Using original text as resolved for now
            )
            citations.append(cite)

    # Process each section
    process_section_citations(
        combined.majority_opinion_citations, OpinionSection.majority
    )
    process_section_citations(
        combined.concurrent_opinion_citations, OpinionSection.concurrent
    )
    process_section_citations(combined.dissenting_citations, OpinionSection.dissenting)

    return cluster, citations


def from_sql_models(
    cluster: OpinionClusterExtraction,
) -> CombinedResolvedCitationAnalysis:
    """Convert SQL models back to a CombinedResolvedCitationAnalysis.

    Args:
        cluster: OpinionClusterExtraction instance with citations relationship loaded
    """
    # Sort citations by section
    majority_citations = []
    concurrent_citations = []
    dissenting_citations = []

    for citation in cluster.citations:
        resolved_citation = CitationResolved(
            page_number=citation.page_number or 1,  # Default to 1 if None
            citation_text=citation.citation_text,
            reasoning=citation.reasoning or "",  # Default to empty string if None
            type=citation.citation_type,
            treatment=citation.treatment
            or CitationTreatment.neutral,  # Default to neutral if None
            relevance=citation.relevance or 1,  # Default to 1 if None
            resolved_opinion_cluster=citation.resolved_opinion_cluster,
            resolved_statute_code_regulation_rule=None,  # Not implemented yet
        )

        if citation.section == OpinionSection.majority:
            majority_citations.append(resolved_citation)
        elif citation.section == OpinionSection.concurrent:
            concurrent_citations.append(resolved_citation)
        elif citation.section == OpinionSection.dissenting:
            dissenting_citations.append(resolved_citation)

    return CombinedResolvedCitationAnalysis(
        date=cluster.date_filed.strftime("%Y-%m-%d"),
        cluster_id=cluster.cluster_id,
        brief_summary=cluster.brief_summary,
        majority_opinion_citations=majority_citations,
        concurrent_opinion_citations=concurrent_citations,
        dissenting_citations=dissenting_citations,
    )
