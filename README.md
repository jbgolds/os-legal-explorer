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

This project is fully dockerized for easy deployment. Follow these steps to get started:

### Prerequisites

- Docker and Docker Compose installed on your system

### Quick Start

1. Clone the repository
2. Configure environment variables in `.env` file (optional - defaults are provided)
3. Start the services:

```bash
docker-compose up -d
```

This will start the following services:
- API service on port 8000
- PostgreSQL database on port 5432
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

