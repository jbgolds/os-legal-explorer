# PRD & Design Document: Supreme Court Opinion Citation Extraction & Visualization

## 1. Project Overview

This project aims to create a free, publicly accessible dataset derived from Supreme Court opinions (sourced from CourtListener’s database) and to build a web-based visualization tool. The goal is to enable users to explore and analyze citation networks between judicial opinions and various legal sources using an interactive graph interface implemented with d3.js, htmx, and FastAPI.

## 2. Objectives and Scope

- **Primary Goal:**  
  - Create a free dataset of citation mappings extracted from Supreme Court opinions.
  - Develop an interactive visualization website using d3.js to display these citation maps.

- **Scope:**  
  - **Initial Focus:** Supreme Court opinions.  
  - **Future Expansion:** Extend to additional types of opinions and legal documents.
  - **Data Source:** A local copy of CourtListener’s database.

## 3. Data Source & Input

- **Database Query:**  
  The following SQL query is used to extract opinion data (with filters for court, date, and published status):

  ```sql
  SELECT  
      so.cluster_id as cluster_id, 
      so.type as so_type, 
      so.id as so_id, 
      so.page_count as so_page_count, 
      so.html_with_citations as so_html_with_citations, 
      so.html as so_html, 
      so.plain_text as so_plain_text, 
      soc.case_name as cluster_case_name,
      soc.date_filed as soc_date_filed,
      soc.citation_count as soc_citation_count,
      sd.court_id as court_id,
      sd.docket_number as sd_docket_number,
      sc.full_name as court_name
  FROM search_opinion so 
  LEFT JOIN search_opinioncluster soc ON so.cluster_id = soc.id
  LEFT JOIN search_docket sd ON soc.docket_id = sd.id
  LEFT JOIN search_court sc ON sd.court_id = sc.id
  {filter_clause}
  ORDER BY soc.date_filed DESC
  ```

- **Input Consistency:**  
  - The majority of the opinion texts are consistent.
  - Occasional JSON formatting issues may occur; when detected, the API request should be retried.

## 4. Citation Extraction Process

- **Extraction Methodology:**  
  - Use language models (LLMs) to process the opinion text and extract citations.
  - Leverage structured outputs to capture detailed citation information.

- **Normalization:**  
  - Integrate eyecite for citation normalization and lookup.

- **Target Citation Types:**  
  The following citation types are defined:
  - `judicial_opinion`
  - `statutes_codes_regulations`
  - `constitution`
  - `administrative_agency_ruling`
  - `congressional_report`
  - `external_submission`
  - `electronic_resource`
  - `law_review`
  - `legal_dictionary`
  - `other`

## 5. Data Model & Structured Outputs

### 5.1. Opinion & Citation Attributes

- **Opinion Sections & Types:**  
  - **OpinionSection:**  
    - Majority, Concurring, Dissenting  
  - **OpinionType:**  
    - Majority, Concurring, Dissenting, Seriatim, Unknown

- **Citation Model:**  
  Each extracted citation includes:
  - **page_number:** Page of occurrence.
  - **citation_text:** Full text of the citation.
  - **reasoning:** A 2–4 sentence explanation of the citation’s context and relevance.
  - **type:** Citation type (as per the enum defined above).
  - **treatment:** How the citation is used (POSITIVE, NEGATIVE, CAUTION, or NEUTRAL).
  - **relevance:** A numerical score (1–4) indicating citation importance.

- **Citation Analysis Model:**  
  Aggregates the citation data for a given opinion:
  - **date:** Publication date (YYYY-MM-DD).
  - **brief_summary:** A 3–5 sentence summary of the core holding.
  - **majority_opinion_citations:** List of citations from the majority opinion.
  - **concurring_opinion_citations:** List of citations from concurring opinions.
  - **dissenting_citations:** List of citations from dissenting opinions.

*Note:* Pydantic models (or similar) will be used to validate and enforce the structure of these outputs.

## 6. Graph Database Design

- **Storage:**  
  - Use Neo4j to store the graph of judicial opinions and their citations.

- **Mapping Relationships:**  
  - Nodes represent:
    - **Judicial Opinions:** The source opinions (always of type `judicial_opinion`).
    - **Citations:** Represented by various types (any enum value from `CitationType`).
  - Relationships will link a judicial opinion node to its cited sources, categorized by the citation type.
  
- **Simplified Approach (KISS):**  
  - While additional "sources" (like o3-generated maps) were considered, they are excluded from the current design to maintain simplicity.

## 7. Visualization & User Interface

- **Frontend Technology:**  
  - **d3.js:** To build interactive, dynamic citation maps.
  - **htmx:** For seamless HTML interactions.

- **Backend Technology:**  
  - **FastAPI:** To serve the backend API, handle data retrieval, and support the dynamic front end.

- **User Experience:**  
  - The website will allow users to interact with the citation graph—filtering and exploring relationships between opinions and citations.
  - The visualization will help legal researchers and interested parties navigate the complex network of judicial citations.

## 8. Error Handling & Validation

- **Error Handling:**  
  - Retry mechanisms will be in place for handling JSON formatting issues during API requests.
  
- **Data Validation:**  
  - Manual verification of the extracted citations will be conducted to ensure the accuracy and relevance of the data.

## 9. Future Considerations

- **Expansion of Data Sources:**  
  - Although the current implementation focuses on Supreme Court opinions, the system is designed with the flexibility to incorporate additional judicial opinions in the future.

- **Additional Features:**  
  - Future iterations may consider integrating multiple sources or automated verification processes. For now, the focus remains on a clear, maintainable pipeline.

## 10. Conclusion

This document outlines the design and objectives for the Supreme Court Opinion Citation Extraction and Visualization project. The system is designed to:
- Extract and normalize citation data from CourtListener’s database using LLMs.
- Structure the extracted data with robust models.
- Store citation mappings in a Neo4j graph database.
- Provide an interactive visualization interface for users via d3.js, htmx, and FastAPI.

The architecture is intentionally kept simple for the initial implementation, with opportunities for expansion and additional features in the future.

