psql --command "\COPY (
    select  
	so.cluster_id as cluster_id, 
	so.type as so_type, 
	so.id as so_id, 
	so.page_count as so_page_count, 
	so.plain_text as so_plain_text, 
	so.html as so_html, 
	so.html_with_citations as so_html_with_citations, 
	so.xml_harvard as so_xml_harvard,
	so.html_columbia as so_html_columbia,
	so.html_anon_2020 as so_html_anon_2020,
	so.html_lawbox as so_html_lawbox,
	soc.case_name as cluster_case_name,
	soc.citation_count as soc_citation_count,
	soc.nature_of_suit as soc_nature_of_suit,
	soc.scdb_decision_direction as soc_scdb_decision_direction,
	soc.scdb_votes_majority as soc_scdb_votes_majority,
	soc.scdb_votes_minority as soc_scdb_votes_minority,
	soc.date_filed as soc_date_filed,
	sd.nature_of_suit as docker_nature_of_suit,
	sd.case_name as docker_case_name,
	sd.id,
	sd.docket_number as sd_docket_number,
 	sd.docket_number_core as sd_docket_number_Core
	from search_opinion so 
	left join search_opinioncluster soc on so.cluster_id = soc.id
	left join search_docket sd on soc.docket_id = sd.id 
where 
	sd.court_id = 'scotus'
	and soc.precedential_status = 'Published'
) TO 'query_with_date.csv' WITH (
    FORMAT csv,
    ENCODING utf8,
    ESCAPE '\\',
    FORCE_QUOTE *,
    HEADER
); " --host "localhost" --username "courtlistener" --dbname "courtlistener"