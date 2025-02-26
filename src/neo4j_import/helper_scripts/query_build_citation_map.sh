#!/bin/bash

# I have no idea why one is in python and the other is in bash
# This is a bash script that queries the PostgreSQL database for the citation map


# PostgreSQL connection details
DB_NAME="courtlistener"
DB_USER="courtlistener"  # default PostgreSQL user, change if needed
DB_HOST="localhost"
OUTPUT_FILE="cl_citation_map.csv"

psql --command "\COPY (
   SELECT 
    so.cluster_id as cited_cluster_id,
    so2.cluster_id as citing_cluster_id,
    'cl_citation_map' as source
FROM search_opinionscited cited
LEFT JOIN search_opinion so ON cited.cited_opinion_id = so.id
LEFT JOIN search_opinion so2 ON cited.citing_opinion_id = so2.id
LEFT JOIN search_opinioncluster soc1 ON so.cluster_id = soc1.id
LEFT JOIN search_opinioncluster soc2 ON so2.cluster_id = soc2.id
WHERE soc1.precedential_status = 'Published'
AND soc2.precedential_status = 'Published'
) TO '$OUTPUT_FILE' WITH (
    FORMAT csv,
    ENCODING utf8,
    FORCE_QUOTE *,
    HEADER
);" --host "$DB_HOST" --username "$DB_USER" --dbname "$DB_NAME"