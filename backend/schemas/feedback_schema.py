from typing import List
from pydantic import BaseModel, Field


class TagFeedbackItem(BaseModel):
    tag: str
    status: str


class FeedbackRequest(BaseModel):
    session_id: str
    reviewed_tags: List[TagFeedbackItem]
    new_tags: List[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    message: str
    session_id: str
    saved_reviewed_tags: int
    saved_new_tags: int
    embedding_path: str