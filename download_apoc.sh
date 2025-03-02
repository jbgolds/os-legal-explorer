#!/bin/bash

# Create a temporary directory
mkdir -p temp_download

# Download the APOC plugin
echo "Downloading APOC plugin..."
curl -L https://github.com/neo4j/apoc/releases/download/2025.01.0/apoc-2025.01.0-core.jar -o temp_download/apoc-2025.01.0-core.jar

# Get the Neo4j container ID
CONTAINER_ID=$(docker ps -qf "name=neo4j")

if [ -z "$CONTAINER_ID" ]; then
  echo "Neo4j container not found. Make sure it's running."
  exit 1
fi

# Copy the plugin to the container's plugins directory
echo "Copying APOC plugin to Neo4j container..."
docker cp temp_download/apoc-2025.01.0-core.jar $CONTAINER_ID:/var/lib/neo4j/plugins/

# Restart the Neo4j container to apply changes
echo "Restarting Neo4j container to apply changes..."
docker restart $CONTAINER_ID

# Clean up
rm -rf temp_download

echo "APOC plugin installation completed!" 