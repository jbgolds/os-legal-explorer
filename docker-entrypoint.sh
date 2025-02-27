#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! pg_isready -h ${DB_HOST} -p ${DB_PORT} -U ${DB_USER}; do
  sleep 2
done
echo "PostgreSQL is ready!"

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to be ready..."
while ! curl -s "http://${NEO4J_USER}:${NEO4J_PASSWORD}@neo4j:7474/browser/" > /dev/null; do
  sleep 2
done
echo "Neo4j is ready!"

# Run database migrations or initialization if needed
# python -m src.some_migration_script

# Start the application
exec "$@" 