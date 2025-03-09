#!/bin/bash

#############
#
#
#      NOTE THIS FILE MUST BE WITHIN neo4j_import/ in order to get mounted
#      TO THE DOCKER CONTAINER
# 
#
#########

# Set password from environment variable
 # TODO Hardcoded variabl
# Create timestamp for the bad entries log
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Run the import with correct syntax
/var/lib/neo4j/bin/neo4j-admin database import full neo4j \
    --nodes=/var/lib/neo4j/import/opinions.header,/var/lib/neo4j/import/cl_opinion_cluster_nodes.csv \
    --relationships=/var/lib/neo4j/import/citations.header,/var/lib/neo4j/import/cl_citation_map_source.csv \
    --overwrite-destination \
    --verbose \
    --multiline-fields=true \
    --bad-tolerance=90000000 \
    --skip-bad-relationships=true \
    --report-file=/var/lib/neo4j/import/import_report_${TIMESTAMP}.log

# Create indexes (after database is running)
/var/lib/neo4j/bin/cypher-shell -u neo4j -p $NEO4J_PASSWORD "
CREATE CONSTRAINT opinion_cluster_id IF NOT EXISTS
FOR (o:Opinion) REQUIRE o.cluster_id IS UNIQUE;

CREATE INDEX opinion_date_filed IF NOT EXISTS
FOR (o:Opinion) ON (o.soc_date_filed);

CREATE INDEX opinion_court_id IF NOT EXISTS
FOR (o:Opinion) ON (o._court_id);

CREATE RANGE INDEX opinion_cluster_id_range IF NOT EXISTS
FOR (o:Opinion) ON (o.cluster_id);
"

echo "Import completed! Created indexes."

# Get count of imported opinions and relationships
echo "Counting imported opinions and citations..."
/var/lib/neo4j/bin/cypher-shell -u neo4j -p $NEO4J_PASSWORD "
MATCH (o:Opinion) 
RETURN count(o) as opinion_count;

MATCH ()-[r]->() 
RETURN type(r) as relationship_type, count(r) as relationship_count;" 