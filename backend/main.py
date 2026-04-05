from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.database import Base, engine
from backend.db import models
from backend.api.predict import router as predict_router
from backend.api.feedback import router as feedback_router
from backend.api.training import router as training_router
from backend.core.model_manager import model_manager
from backend.services.automation_service import automation_service


app = FastAPI(
    title="Human-in-the-Loop Image Tagging System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://3.135.205.178:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    model_manager.load_all_models()
    automation_service.start()


@app.on_event("shutdown")
def on_shutdown():
    automation_service.stop()


@app.get("/")
def root():
    return {
        "message": "Human-in-the-Loop Image Tagging backend is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "adaptive_model_ready": model_manager.adaptive_ready,
        "automation_running": automation_service.started
    }


app.include_router(predict_router)
app.include_router(feedback_router)
app.include_router(training_router)