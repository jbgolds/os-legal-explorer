#!/bin/bash
set -e

# Basic check for PostgreSQL
echo "Checking connection to PostgreSQL..."
pg_isready -h ${DB_HOST} -p ${DB_PORT} -U ${DB_USER} || echo "PostgreSQL not yet accessible - API will retry connections as needed"

# Basic check for Neo4j
echo "Checking connection to Neo4j..."
curl -s --head "http://${NEO4J_USER}:${NEO4J_PASSWORD}@neo4j:7474/browser/" > /dev/null || echo "Neo4j not yet accessible - API will retry connections as needed"

# Start the application
echo "Starting the application..."
exec "$@" 