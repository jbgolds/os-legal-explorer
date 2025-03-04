import logging
import json
import os
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, EmailStr
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/feedback",
    tags=["feedback"],
    responses={404: {"description": "Not found"}},
)


# Models
class MissingCitationFeedback(BaseModel):
    cluster_id: str
    expected_citation: str
    description: str
    email: Optional[EmailStr] = None
    submitted_at: datetime = datetime.now()


class MissingOpinionFeedback(BaseModel):
    cluster_id: str
    description: str
    email: Optional[EmailStr] = None
    submitted_at: datetime = datetime.now()


class GeneralFeedback(BaseModel):
    feedback_type: str  # "bug", "feature", "other"
    description: str
    email: Optional[EmailStr] = None
    submitted_at: datetime = datetime.now()


# Ensure feedback directory exists
FEEDBACK_DIR = os.path.join("data", "feedback")
os.makedirs(FEEDBACK_DIR, exist_ok=True)


# Helper function to save feedback to JSON file
def save_feedback(feedback_type: str, data: dict):
    """Save feedback data to a JSON file."""
    filename = os.path.join(
        FEEDBACK_DIR, f"{feedback_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(filename, "w") as f:
        json.dump(data, f, default=str)
    return filename


# Missing citation feedback endpoint
@router.post("/citation/missing", status_code=201)
async def report_missing_citation(feedback: MissingCitationFeedback):
    """
    Report a missing citation for a case.

    Parameters:
    - cluster_id: The ID of the case
    - expected_citation: The citation that should be included
    - description: Description of the issue
    - email: Optional email for follow-up
    """
    try:
        # Save feedback to file
        filename = save_feedback("missing_citation", feedback.dict())
        logger.info(f"Saved missing citation feedback to {filename}")

        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error saving missing citation feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


# Missing opinion feedback endpoint
@router.post("/opinion/missing", status_code=201)
async def report_missing_opinion(feedback: MissingOpinionFeedback):
    """
    Report a missing opinion for a case.

    Parameters:
    - cluster_id: The ID of the case
    - description: Description of the issue
    - email: Optional email for follow-up
    """
    try:
        # Save feedback to file
        filename = save_feedback("missing_opinion", feedback.dict())
        logger.info(f"Saved missing opinion feedback to {filename}")

        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error saving missing opinion feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


# General feedback endpoint
@router.post("/general", status_code=201)
async def submit_general_feedback(feedback: GeneralFeedback):
    """
    Submit general feedback about the platform.

    Parameters:
    - feedback_type: Type of feedback ("bug", "feature", "other")
    - description: Description of the feedback
    - email: Optional email for follow-up
    """
    try:
        # Save feedback to file
        filename = save_feedback("general", feedback.dict())
        logger.info(f"Saved general feedback to {filename}")

        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error saving general feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")
