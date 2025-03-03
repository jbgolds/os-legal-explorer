import os
from neo4j import GraphDatabase, basic_auth


def clear_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    with driver.session() as session:
        # Count nodes before deletion
        result = session.run("MATCH (n) RETURN count(n) as count")
        before = result.single()["count"]
        print(f"Nodes before deletion: {before}")

        # Delete all nodes and relationships
        session.run("MATCH (n) DETACH DELETE n")

        # Count nodes after deletion
        result = session.run("MATCH (n) RETURN count(n) as count")
        after = result.single()["count"]
        print(f"Nodes after deletion: {after}")
    driver.close()


if __name__ == "__main__":
    # Configure your Neo4j connection. Default values are provided for local development.
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "courtlistener")

    print("Starting Neo4j database clearance...")
    clear_neo4j(uri, user, password)
    print("Neo4j database cleared.")
