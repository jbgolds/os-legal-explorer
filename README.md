# Goal of this project
I want to create a very interactive website using d3.js, htmx, and a python backend preferrably something simple, like fastapi. 

- There will be no accounts, just a frontend with some different views, probably tabs work for now.
- We will get more into the visualizations later.


I have a local postgres database running with a copy `[courtlistenerof](http://courtlistener.com)'s dumps of all of their data. With specifically the Court Opinions text, I will be mappings of citations found in these cases to one another, while staying consistent with courtlistener's data to hopefully integrate one day. I've built a pipeline already to use LLM's to extract the citation's treatment and other metadata out of all citations in the opinion's text. I also have a neo4j database where I'd like to build a graph database of these citation representations. I have models built out for all three of these datatypes to use in python. I have a VPS with 64GB of RAM and 2TB SSD, 16 dedicated server cores, so we have plenty of processing power.

Needs to be done:
- connect the pipes and start getting citations loaded in! 
- build out a python web backend to manage two things:
    1. the pipeline between the csv export of the database table, calling LLM, resolving opinion cluster ID, and entering into neo4j.
    2. web app using a python + htmx + d3.js + tailwindcss to create a nice looking website to display our rich graphs of information.
    

## Docker Setup

This project is dockerized for easy deployment with Neo4j and the API service. It assumes PostgreSQL is running separately on your host machine.

### Prerequisites

- Docker and Docker Compose installed on your system
- PostgreSQL running on your host machine
- PostgreSQL database created for the project

### PostgreSQL Setup

Before starting the Docker services, make sure your PostgreSQL server:
1. Is running and accessible
2. Has a database named `courtlistener` (or update the DB_NAME in docker-compose.yml)
3. Has a user `courtlistener` with password `postgrespassword` (or update DB_USER and DB_PASSWORD in docker-compose.yml)
4. Allows connections from Docker containers (check pg_hba.conf)

### Quick Start

1. Clone the repository
2. Configure environment variables in docker-compose.yml to match your PostgreSQL setup
3. Start the services:

```bash
docker-compose up -d
```

This will start the following services:
- API service on port 8000
- Neo4j database on ports 7474 (HTTP) and 7687 (Bolt)

### Accessing Services

- FastAPI documentation: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474/browser/

### Development

For development, the code is mounted as a volume, so changes will be reflected immediately.
When you make changes to the API code, the server will auto-reload.

### Stopping Services

```bash
docker-compose down
```

To remove all data (volumes):

```bash
docker-compose down -v
```

## Simple Deployment Guide

This project can be easily deployed on a VPS with Docker. Here's a straightforward approach:

### Prerequisites

- A VPS with Docker and Docker Compose installed
- PostgreSQL installed and running on the VPS or accessible from the VPS
- Git installed
- At least 2GB RAM recommended

### PostgreSQL Setup on VPS

If PostgreSQL is running on the same VPS:

1. Install PostgreSQL if not already installed:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

2. Create a database and user:
```bash
sudo -u postgres psql
postgres=# CREATE DATABASE courtlistener;
postgres=# CREATE USER courtlistener WITH PASSWORD 'postgrespassword';
postgres=# GRANT ALL PRIVILEGES ON DATABASE courtlistener TO courtlistener;
postgres=# \q
```

3. Update PostgreSQL configuration to allow connections:
```bash
sudo nano /etc/postgresql/*/main/pg_hba.conf
```
Add this line (adjust for your network setup):
```
host    all             all             172.17.0.0/16           md5
```
Then restart PostgreSQL:
```bash
sudo systemctl restart postgresql
```

### Deployment Steps

1. Clone the repository on your VPS:
```bash
git clone <your-repo-url>
cd os-legal-explorer
```

2. Update docker-compose.yml to connect to your PostgreSQL server:
```yaml
# In the api service section:
environment:
  - DB_HOST=your-postgres-host-or-ip
  - DB_PORT=5432
  - DB_USER=courtlistener
  - DB_PASSWORD=your-secure-password
  - DB_NAME=courtlistener
  # ... other environment variables ...
```

3. Make the entrypoint script executable:
```bash
chmod +x docker-entrypoint.sh
```

4. Start the services:
```bash
docker-compose up -d
```

This will start:
- The API service on port 8000
- Neo4j database on ports 7474 (HTTP) and 7687 (Bolt)

### Importing Data

If you have CourtListener data:

1. For PostgreSQL data (since PostgreSQL is running separately):
```bash
# For SQL dumps (run directly on your PostgreSQL server)
psql -U courtlistener -d courtlistener -f /path/to/your/dump.sql

# For CSV files (adjust paths as needed)
# Example using psql's \copy command
psql -U courtlistener -d courtlistener
courtlistener=# \copy table_name FROM '/path/to/your/data.csv' WITH CSV HEADER;
```

2. Use the API pipeline endpoints to process data:
- Access the API docs at http://your-vps-ip:8000/docs
- Use the pipeline endpoints to extract, process and load data

### Monitoring and Maintenance

- View logs with `docker-compose logs -f`
- Restart services with `docker-compose restart`
- Stop all services with `docker-compose down`

