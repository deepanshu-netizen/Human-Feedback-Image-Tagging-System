import json
import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from backend.db.models import (
    SessionRecord,
    TagFeedbackRecord,
    TrainingRunRecord,
    TrainingRunSessionRecord,
)


def create_training_run(
    db: Session,
    status: str,
    forced: bool,
    model_version: str | None = None,
    num_sessions_used: int = 0,
    summary_json: str | None = None,
    error_message: str | None = None,
    completed_at=None,
):
    run_id = str(uuid.uuid4())

    record = TrainingRunRecord(
        run_id=run_id,
        status=status,
        forced=1 if forced else 0,
        model_version=model_version,
        num_sessions_used=num_sessions_used,
        summary_json=summary_json,
        error_message=error_message,
        completed_at=completed_at,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def update_training_run(
    db: Session,
    run_id: str,
    status: str | None = None,
    model_version: str | None = None,
    num_sessions_used: int | None = None,
    summary_json: str | None = None,
    error_message: str | None = None,
    completed_at=None,
):
    record = (
        db.query(TrainingRunRecord)
        .filter(TrainingRunRecord.run_id == run_id)
        .first()
    )

    if record is None:
        return None

    if status is not None:
        record.status = status
    if model_version is not None:
        record.model_version = model_version
    if num_sessions_used is not None:
        record.num_sessions_used = num_sessions_used
    if summary_json is not None:
        record.summary_json = summary_json
    if error_message is not None:
        record.error_message = error_message
    if completed_at is not None:
        record.completed_at = completed_at

    db.commit()
    db.refresh(record)
    return record


def add_sessions_to_training_run(
    db: Session,
    run_id: str,
    session_ids: list[str]
):
    rows = []
    for session_id in session_ids:
        row = TrainingRunSessionRecord(
            run_id=run_id,
            session_id=session_id
        )
        db.add(row)
        rows.append(row)

    db.commit()
    return rows


def get_latest_completed_training_run(db: Session):
    return (
        db.query(TrainingRunRecord)
        .filter(TrainingRunRecord.status == "completed")
        .order_by(TrainingRunRecord.created_at.desc())
        .first()
    )


def get_session_ids_for_training_run(db: Session, run_id: str):
    rows = (
        db.query(TrainingRunSessionRecord)
        .filter(TrainingRunSessionRecord.run_id == run_id)
        .all()
    )
    return [row.session_id for row in rows]


def create_session_record(
    db: Session,
    session_id: str,
    image_path: str,
    embedding_path: str | None,
    extracted_tags: list[str],
    new_tags_added: list[str]
):
    record = SessionRecord(
        session_id=session_id,
        image_path=image_path,
        embedding_path=embedding_path,
        extracted_tags=json.dumps(extracted_tags, ensure_ascii=False),
        new_tags_added=json.dumps(new_tags_added, ensure_ascii=False),
        num_generated_tags=len(extracted_tags),
        num_new_tags=len(new_tags_added)
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def create_tag_feedback_records(
    db: Session,
    session_id: str,
    image_path: str,
    embedding_path: str | None,
    tag_status_pairs: list[tuple[str, str]]
):
    records = []
    for tag, status in tag_status_pairs:
        row = TagFeedbackRecord(
            session_id=session_id,
            image_path=image_path,
            embedding_path=embedding_path,
            generated_tag=tag,
            human_status=status
        )
        db.add(row)
        records.append(row)

    db.commit()
    return records


def get_session_by_session_id(db: Session, session_id: str):
    return (
        db.query(SessionRecord)
        .filter(SessionRecord.session_id == session_id)
        .first()
    )


def update_session_feedback(
    db: Session,
    session_record: SessionRecord,
    embedding_path: str | None,
    new_tags_added: list[str]
):
    session_record.embedding_path = embedding_path
    session_record.new_tags_added = json.dumps(new_tags_added, ensure_ascii=False)
    session_record.num_new_tags = len(new_tags_added)

    db.commit()
    db.refresh(session_record)
    return session_record