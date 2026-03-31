import json
from PIL import Image
from fastapi import APIRouter, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from pydantic import ValidationError

from backend.db.database import get_db
from backend.db.crud import get_session_by_session_id
from backend.schemas.feedback_schema import FeedbackRequest, FeedbackResponse
from backend.services.feedback_service import save_feedback


router = APIRouter()


@router.post("/submit-feedback", response_model=FeedbackResponse)
def submit_feedback(
    payload_json: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        try:
            payload_dict = json.loads(payload_json)
            payload = FeedbackRequest(**payload_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in payload_json.")
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=json.loads(e.json()))

        session_record = get_session_by_session_id(db, payload.session_id)
        if session_record is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        if not session_record.image_path:
            raise HTTPException(status_code=400, detail="No image path found for this session.")

        try:
            image = Image.open(session_record.image_path).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load image from saved session path: {str(e)}"
            )

        result = save_feedback(
            db=db,
            session_record=session_record,
            image=image,
            reviewed_tags=payload.reviewed_tags,
            new_tags=payload.new_tags
        )

        return {
            "message": "Feedback saved successfully.",
            **result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback save failed: {str(e)}")