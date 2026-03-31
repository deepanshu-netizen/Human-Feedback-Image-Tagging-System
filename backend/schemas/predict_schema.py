from typing import List, Optional
from pydantic import BaseModel


class TagPrediction(BaseModel):
    tag: str
    score: Optional[float] = None


class PredictResponse(BaseModel):
    session_id: str
    ram_tags: List[str]
    adaptive_tags: List[TagPrediction]
    combined_tags: List[str]