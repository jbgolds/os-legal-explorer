


# Make command to access the logs from a specific job 
see-logs:
	docker exec os-legal-explorer-api-1 cat /tmp/$(job_id).log


# Make command to access the logs from a specific job 

copy-logs:
	for file in $(docker exec os-legal-explorer-api-1 ls /tmp | grep -E '\.json|\.csv'); do docker cp os-legal-explorer-api-1:/tmp/$file debug_files/; done

repomix:
	repomix --ignore "*.csv,*.json,uv.lock,.venv/,prd.md"

# for file in $(docker exec os-legal-explorer-api-1 ls /tmp | grep -E '\.json|\.csv'); do docker cp os-legal-explorer-api-1:/tmp/$file debug_files/; done
# clear the database

clear-neo4j:
	docker volume rm os-legal-explorer_neo4j_data
# MATCH (n) DETACH DELETE n
# CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *