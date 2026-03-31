import json
from datetime import datetime

from backend.db.database import SessionLocal
from backend.db.crud import (
    create_training_run,
    update_training_run,
    add_sessions_to_training_run,
    get_latest_completed_training_run,
    get_session_ids_for_training_run,
)
from backend.db.models import SessionRecord, TagFeedbackRecord


def create_training_run_record(status: str, forced: bool):
    db = SessionLocal()
    try:
        record = create_training_run(
            db=db,
            status=status,
            forced=forced,
            completed_at=None
        )
        return record.run_id
    finally:
        db.close()


def complete_training_run_record(
    run_id: str,
    status: str,
    model_version: str | None,
    session_ids: list[str],
    summary: dict,
    error_message: str | None = None,
):
    db = SessionLocal()
    try:
        update_training_run(
            db=db,
            run_id=run_id,
            status=status,
            model_version=model_version,
            num_sessions_used=len(session_ids),
            summary_json=json.dumps(summary, ensure_ascii=False),
            error_message=error_message,
            completed_at=datetime.utcnow(),
        )

        if session_ids:
            add_sessions_to_training_run(
                db=db,
                run_id=run_id,
                session_ids=session_ids
            )
    finally:
        db.close()


def get_latest_completed_run_info():
    db = SessionLocal()
    try:
        run = get_latest_completed_training_run(db)
        if run is None:
            return None

        session_ids = get_session_ids_for_training_run(db, run.run_id)

        return {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "completed_at": run.completed_at,
            "model_version": run.model_version,
            "session_ids": session_ids,
        }
    finally:
        db.close()


def get_usable_feedback_session_ids():
    """
    Returns session_ids that:
    - exist in sessions table
    - have an embedding_path
    - have at least one tag_feedback row or at least one new tag
    """
    db = SessionLocal()
    try:
        session_rows = db.query(SessionRecord).all()

        usable_session_ids = []

        for session in session_rows:
            if not session.embedding_path:
                continue

            has_tag_feedback = (
                db.query(TagFeedbackRecord)
                .filter(TagFeedbackRecord.session_id == session.session_id)
                .first()
                is not None
            )

            has_new_tags = False
            if session.new_tags_added:
                try:
                    parsed = json.loads(session.new_tags_added)
                    has_new_tags = isinstance(parsed, list) and len(parsed) > 0
                except Exception:
                    has_new_tags = bool(str(session.new_tags_added).strip())

            if has_tag_feedback or has_new_tags:
                usable_session_ids.append(session.session_id)

        return usable_session_ids
    finally:
        db.close()