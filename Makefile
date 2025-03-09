


# Make command to access the logs from a specific job 
see-logs:
	docker exec os-legal-explorer-api-1 cat /tmp/$(job_id).log


# Make command to access the logs from a specific job 

copy-logs:
	LATEST_DATE_HOUR=$$(docker exec os-legal-explorer-api-1 ls -la /tmp | grep -E '\.json|\.csv' | awk '{print $$9}' | sed 's/.*_\([0-9]\{8\}_[0-9]\{4\}\)[0-9]\{2\}\..*/\1/' | sort -r | head -1) && \
	echo "Copying files with date/hour: $$LATEST_DATE_HOUR" && \
	for file in $$(docker exec os-legal-explorer-api-1 ls /tmp | grep -E "\.json|\.csv" | grep "$$LATEST_DATE_HOUR"); do \
		echo "Copying $$file" && \
		docker cp os-legal-explorer-api-1:/tmp/$$file debug_files/; \
	done


copy-all-logs:
	for file in $$(docker exec os-legal-explorer-api-1 ls /tmp | grep -E '\.json|\.csv'); do \
		echo "Copying $$file" && \
		docker cp os-legal-explorer-api-1:/tmp/$$file debug_files/; \
	done


repomix:
	repomix --ignore "*.csv,*.json,uv.lock,.venv/,prd.md"

# for file in $(docker exec os-legal-explorer-api-1 ls /tmp | grep -E '\.json|\.csv'); do docker cp os-legal-explorer-api-1:/tmp/$file debug_files/; done
# clear the database

clear-neo4j:
	docker volume rm os-legal-explorer_neo4j_data
# MATCH (n) DETACH DELETE n
# CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *


# put old neo4j.dumps into neo4j_backups/ and run to restore
restore-neo4j:
	docker compose run --rm --entrypoint="" neo4j bash -c "neo4j-admin database load --from-path=/backups --overwrite-destination neo4j"