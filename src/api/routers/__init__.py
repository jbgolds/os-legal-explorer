# This file marks the directory as a Python package
# Import and expose all routers
from src.api.routers.clusters import router as clusters_router
from src.api.routers.feedback import router as feedback_router
from src.api.routers.network import router as network_router
from src.api.routers.search import router as search_router
from src.api.routers.stats import router as stats_router

__all__ = ["search_router", "clusters_router", "feedback_router", "network_router", "stats_router"]
