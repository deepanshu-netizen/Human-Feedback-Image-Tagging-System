from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from PIL import Image
import io

from backend.db.database import get_db
from backend.db.crud import create_session_record
from backend.schemas.predict_schema import PredictResponse
from backend.services.inference_service import run_prediction, save_uploaded_image


router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict_tags(
    file: UploadFile = File(...),
    num_tags: int = Form(8),
    db: Session = Depends(get_db)
):
    try:
        num_tags = max(1, min(int(num_tags), 20))

        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        prediction_result = run_prediction(image, max_tags=num_tags)

        image_path = save_uploaded_image(
            image=image,
            session_id=prediction_result["session_id"]
        )

        create_session_record(
            db=db,
            session_id=prediction_result["session_id"],
            image_path=image_path,
            embedding_path=None,
            extracted_tags=prediction_result["combined_tags"],
            new_tags_added=[]
        )

        return prediction_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")