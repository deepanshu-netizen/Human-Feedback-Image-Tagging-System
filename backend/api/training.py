from fastapi import APIRouter, HTTPException, Query

from backend.core.model_manager import model_manager
from backend.services.training_job_service import (
    start_retraining_job,
    get_retraining_status,
)


router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/retrain/start")
def start_retraining(force: bool = Query(False)):
    try:
        return start_retraining_job(force=force)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not start retraining: {str(e)}")


@router.get("/retrain/status")
def retrain_status():
    try:
        status_data = get_retraining_status()
        status_data["adaptive_model_ready"] = model_manager.adaptive_ready
        return status_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read retraining status: {str(e)}")


@router.post("/retrain/reload-model")
def reload_adaptive_model():
    try:
        model_manager.reload_adaptive_model()
        return {
            "message": "Adaptive model reload attempted.",
            "adaptive_model_ready": model_manager.adaptive_ready
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not reload model: {str(e)}")