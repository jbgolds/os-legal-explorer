from fastapi import Request, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.api.services.db_utils import get_opinion_by_id
from src.api.services.opinion_service import get_opinion_citations
from src.api.services.pipeline_service import get_jobs

# Create router
router = APIRouter()

# Configure templates
templates = Jinja2Templates(directory="src/frontend/templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/api")
async def api_root():
    return {
        "message": "Welcome to the OS Legal Explorer API",
        "docs": "/docs",
        "version": "0.1.0",
    }


@router.get("/pipeline", response_class=HTMLResponse)
async def pipeline_tasks(request: Request):
    # Get all jobs, sorted by newest first, limit to last 50
    tasks = get_jobs(None, limit=50)
    return templates.TemplateResponse(
        "pipeline-tasks.html", {"request": request, "tasks": tasks}
    )


@router.get("/case/{cluster_id}", response_class=HTMLResponse)
async def get_case(request: Request, cluster_id: int):
    # Get case details from your existing API
    case = await get_opinion_by_id(cluster_id)

    # Get citation counts
    outgoing = await get_opinion_citations(cluster_id, direction="outgoing")
    incoming = await get_opinion_citations(cluster_id, direction="incoming")

    return templates.TemplateResponse(
        "case-view.html",
        {
            "request": request,
            "case": case,
            "outgoing_count": len(outgoing),
            "incoming_count": len(incoming),
        },
    )


@router.get("/case/{cluster_id}/citations", response_class=HTMLResponse)
async def get_case_citations(
    request: Request, cluster_id: int, direction: str = "outgoing"
):
    # Get citations from your existing API
    citations = await get_opinion_citations(cluster_id, direction=direction)

    return templates.TemplateResponse(
        "citations-list.html", {"request": request, "citations": citations}
    )
