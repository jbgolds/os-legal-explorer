from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .api.routers import cases, search
from .frontend.routers import pages
from .neo4j_db.init_db import init_neo4j

app = FastAPI()

# Initialize Neo4j database
init_neo4j()

# Mount static files
app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")

# Include routers
app.include_router(cases.router)
app.include_router(search.router)
app.include_router(pages.router)

# Templates
templates = Jinja2Templates(directory="src/frontend/templates")
