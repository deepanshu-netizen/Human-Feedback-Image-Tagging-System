from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from datetime import datetime

from backend.db.database import Base


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    image_path = Column(String, nullable=False)
    embedding_path = Column(String, nullable=True)

    extracted_tags = Column(Text, nullable=True)
    new_tags_added = Column(Text, nullable=True)

    num_generated_tags = Column(Integer, default=0, nullable=False)
    num_new_tags = Column(Integer, default=0, nullable=False)


class TagFeedbackRecord(Base):
    __tablename__ = "tag_feedback"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    image_path = Column(String, nullable=False)
    embedding_path = Column(String, nullable=True)

    generated_tag = Column(String, nullable=False)
    human_status = Column(String, nullable=False)


class TrainingRunRecord(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    status = Column(String, nullable=False)  # running / completed / skipped / failed
    forced = Column(Integer, default=0, nullable=False)  # 0 or 1

    model_version = Column(String, nullable=True)
    num_sessions_used = Column(Integer, default=0, nullable=False)

    summary_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)


class TrainingRunSessionRecord(Base):
    __tablename__ = "training_run_sessions"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("training_runs.run_id"), index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False)