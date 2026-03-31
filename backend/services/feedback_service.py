import os
import numpy as np
from sqlalchemy.orm import Session

from backend.config import EMBEDDINGS_DIR, STATUS_CHOICES
from backend.core.model_manager import model_manager
from backend.db.crud import (
    create_tag_feedback_records,
    update_session_feedback,
)
from shared.utils import normalize_image_input, normalize_new_tags


def save_embedding(embedding, session_id: str) -> str:
    embedding_path = os.path.join(EMBEDDINGS_DIR, f"{session_id}.npy")
    np.save(embedding_path, embedding)
    return embedding_path


def normalize_status(status: str) -> str:
    status = str(status).strip().lower()
    if status not in STATUS_CHOICES:
        raise ValueError(f"Invalid status: {status}")
    return status


def save_feedback(
    db: Session,
    session_record,
    image,
    reviewed_tags,
    new_tags
):
    image = normalize_image_input(image)

    clip_encoder = model_manager.get_clip_encoder()
    embedding = clip_encoder.get_image_embedding(image)
    embedding_path = save_embedding(embedding, session_record.session_id)

    tag_status_pairs = []
    for item in reviewed_tags:
        tag = str(item.tag).strip().lower()
        status = normalize_status(item.status)

        if tag:
            tag_status_pairs.append((tag, status))

    normalized_new_tags = normalize_new_tags(",".join(new_tags))

    create_tag_feedback_records(
        db=db,
        session_id=session_record.session_id,
        image_path=session_record.image_path,
        embedding_path=embedding_path,
        tag_status_pairs=tag_status_pairs
    )

    updated_session = update_session_feedback(
        db=db,
        session_record=session_record,
        embedding_path=embedding_path,
        new_tags_added=normalized_new_tags
    )

    return {
        "session_id": updated_session.session_id,
        "saved_reviewed_tags": len(tag_status_pairs),
        "saved_new_tags": len(normalized_new_tags),
        "embedding_path": embedding_path,
    }