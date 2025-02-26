# import asyncio
# import logging
# from typing import List, Dict, Any, Optional, Tuple
# from datetime import datetime
# from neomodel import config, db
# from models import Opinion, CitesRel, CitationType, CitationTreatment
# from src.llm_extraction.models import CombinedResolvedCitationAnalysis

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

### LEGACY BAD IMPLEMENTATION

# class AsyncNeo4jLoader:
#     def __init__(self, uri: str, username: str, password: str):
#         """Initialize the loader with Neo4j connection details"""
#         self.uri = uri
#         config.DATABASE_URL = f"bolt://{username}:{password}@{uri}"
#         self.batch_size = 1000  # Default batch size
#         self._setup_database()

#     def _setup_database(self):
#         """Setup database constraints and indexes"""
#         # Create constraints and indexes if they don't exist
#         with db.transaction:
#             # These are already handled by neomodel's unique_index=True in the Opinion model
#             # but we'll add any additional indexes we need
#             db.cypher_query(
#                 """
#                 CREATE INDEX opinion_date_filed IF NOT EXISTS
#                 FOR (o:Opinion) ON (o.date_filed)
#                 """
#             )
#             db.cypher_query(
#                 """
#                 CREATE INDEX opinion_court_id IF NOT EXISTS
#                 FOR (o:Opinion) ON (o.court_id)
#                 """
#             )

#     async def load_opinions_async(
#         self, opinions_data: List[Dict[str, Any]], batch_size: Optional[int] = None
#     ) -> List[Opinion]:
#         """
#         Asynchronously load opinions in batches.

#         Args:
#             opinions_data: List of dictionaries containing opinion data
#             batch_size: Optional batch size override

#         Returns:
#             List of created Opinion nodes
#         """
#         batch_size = batch_size or self.batch_size
#         results = []

#         # Process in batches
#         for i in range(0, len(opinions_data), batch_size):
#             batch = opinions_data[i : i + batch_size]
#             batch_results = await asyncio.gather(
#                 *[self._create_opinion(data) for data in batch]
#             )
#             results.extend(batch_results)

#             logger.info(
#                 f"Processed batch of {len(batch)} opinions. "
#                 f"Total processed: {len(results)}"
#             )

#         return results

#     async def _create_opinion(self, data: Dict[str, Any]) -> Opinion:
#         """Create a single opinion node"""
#         # Ensure required cluster_id is present
#         if "cluster_id" not in data:
#             raise ValueError(f"Missing required cluster_id in opinion data: {data}")

#         # Convert date string to datetime if present
#         if "date_filed" in data and isinstance(data["date_filed"], str):
#             data["date_filed"] = datetime.strptime(
#                 data["date_filed"], "%Y-%m-%d"
#             ).date()

#         # Use get_or_create to avoid duplicates
#         opinion, created = await asyncio.to_thread(
#             Opinion.get_or_create,
#             cluster_id=data["cluster_id"],
#             **{k: v for k, v in data.items() if k != "cluster_id"},
#         )

#         if created:
#             logger.debug(f"Created new opinion node: {opinion.cluster_id}")
#         else:
#             logger.debug(f"Found existing opinion node: {opinion.cluster_id}")

#         return opinion

#     async def load_citations_async(
#         self,
#         citations_data: List[CombinedResolvedCitationAnalysis],
#         source: str,
#         batch_size: Optional[int] = None,
#     ):
#         """
#         Asynchronously load citation relationships with metadata.

#         Args:
#             citations_data: List of CombinedResolvedCitationAnalysis objects
#             source: Source of the citations (e.g., 'courtlistener', 'ai_enhanced')
#             batch_size: Optional batch size override
#         """
#         batch_size = batch_size or self.batch_size
#         total_processed = 0

#         for citation_analysis in citations_data:
#             # First ensure the citing opinion exists
#             citing_opinion, _ = await asyncio.to_thread(
#                 Opinion.get_or_create,
#                 cluster_id=citation_analysis.cluster_id,
#                 ai_summary=citation_analysis.brief_summary,
#                 date_filed=citation_analysis.date,
#             )

#             # Process all citation types (majority, concurrent, dissenting)
#             sections = [
#                 ("majority", citation_analysis.majority_opinion_citations),
#                 ("concurrent", citation_analysis.concurrent_opinion_citations),
#                 ("dissent", citation_analysis.dissenting_citations),
#             ]

#             for section_name, citations in sections:
#                 for citation in citations:
#                     if citation.resolved_opinion_cluster:
#                         # Create relationship with metadata
#                         rel_props = {
#                             "source": source,
#                             "opinion_section": section_name,
#                             "treatment": citation.treatment,
#                             "relevance": citation.relevance,
#                             "reasoning": citation.reasoning,
#                             "citation_text": citation.citation_text,
#                             "page_number": citation.page_number,
#                             "timestamp": datetime.now(),
#                         }

#                         # Filter out None values
#                         rel_props = {
#                             k: v for k, v in rel_props.items() if v is not None
#                         }

#                         try:
#                             # Get or create the cited opinion
#                             cited_opinion, _ = await asyncio.to_thread(
#                                 Opinion.get_or_create,
#                                 cluster_id=citation.resolved_opinion_cluster,
#                                 type=citation.type,
#                             )

#                             # Create the relationship
#                             await asyncio.to_thread(
#                                 citing_opinion.cites.connect, cited_opinion, rel_props
#                             )

#                             # Increment citation count
#                             await asyncio.to_thread(
#                                 cited_opinion.increment_citation_count
#                             )

#                             total_processed += 1

#                             if total_processed % batch_size == 0:
#                                 logger.info(f"Processed {total_processed} citations")

#                         except Exception as e:
#                             logger.error(f"Error processing citation: {str(e)}")
#                             continue

#         logger.info(f"Completed processing {total_processed} citations")

#     async def load_basic_citations_async(
#         self,
#         citation_pairs: List[Tuple[int, int]],
#         source: str,
#         batch_size: Optional[int] = None,
#     ):
#         """
#         Asynchronously load basic citation relationships (just source and timestamp).
#         Useful for loading the base CourtListener citation network.

#         Args:
#             citation_pairs: List of (citing_cluster_id, cited_cluster_id) tuples
#             source: Source of the citations (e.g., 'courtlistener')
#             batch_size: Optional batch size override
#         """
#         batch_size = batch_size or self.batch_size
#         total_processed = 0

#         # Process in batches
#         for i in range(0, len(citation_pairs), batch_size):
#             batch = citation_pairs[i : i + batch_size]
#             tasks = []

#             for citing_id, cited_id in batch:
#                 task = asyncio.create_task(
#                     self._create_basic_citation(citing_id, cited_id, source)
#                 )
#                 tasks.append(task)

#             # Wait for batch to complete
#             await asyncio.gather(*tasks)
#             total_processed += len(batch)
#             logger.info(f"Processed {total_processed} basic citations")

#     async def _create_basic_citation(self, citing_id: int, cited_id: int, source: str):
#         """Create a basic citation relationship between two opinions"""
#         try:
#             # Get or create both opinions
#             citing_opinion, _ = await asyncio.to_thread(
#                 Opinion.get_or_create, cluster_id=citing_id
#             )
#             cited_opinion, _ = await asyncio.to_thread(
#                 Opinion.get_or_create, cluster_id=cited_id
#             )

#             # Create the basic relationship
#             await asyncio.to_thread(
#                 citing_opinion.cites.connect,
#                 cited_opinion,
#                 {"source": source, "timestamp": datetime.now()},
#             )

#             # Increment citation count
#             await asyncio.to_thread(cited_opinion.increment_citation_count)

#         except Exception as e:
#             logger.error(f"Error creating citation {citing_id}->{cited_id}: {str(e)}")


# # Example usage
# async def main():
#     # Configuration
#     NEO4J_URI = "localhost:7687"
#     NEO4J_USER = "neo4j"
#     NEO4J_PASSWORD = "courtlistener"

#     loader = AsyncNeo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

#     # Example opinion data
#     opinions = [
#         {
#             "cluster_id": 1001,
#             "date_filed": "2020-01-01",
#             "case_name": "Smith v. Jones",
#             "court_id": 5,
#             "court_name": "Supreme Court",
#         },
#         # Add more opinions...
#     ]

#     # Load opinions
#     await loader.load_opinions_async(opinions)

#     # Example citation pairs
#     citation_pairs = [
#         (1001, 1002),
#         (1001, 1003),
#         # Add more pairs...
#     ]

#     # Load basic citations
#     await loader.load_basic_citations_async(citation_pairs, source="courtlistener")


# if __name__ == "__main__":
#     asyncio.run(main())
