
from fastapi import Request
from src.api.main import app, templates
from fastapi.responses import HTMLResponse
from src.api.services.db_utils import get_opinion_by_id
from src.api.services.opinion_service import get_opinion_citations

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/case/{cluster_id}", response_class=HTMLResponse)
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
            "incoming_count": len(incoming)
        }
    )

@app.get("/case/{cluster_id}/citations", response_class=HTMLResponse)
async def get_case_citations(request: Request, cluster_id: int, direction: str = "outgoing"):
    # Get citations from your existing API
    citations = await get_opinion_citations(cluster_id, direction=direction)
    
    return templates.TemplateResponse(
        "citations-list.html", 
        {"request": request, "citations": citations}
    )