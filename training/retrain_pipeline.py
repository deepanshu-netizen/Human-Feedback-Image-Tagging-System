import os
import json
import sqlite3
import argparse
from datetime import datetime, timedelta

from backend.config import DB_PATH, TRAINED_MODELS_DIR
from training.build_dataset import build_dataset
from training.train_classifier import train

from training.training_db import (
    create_training_run_record,
    complete_training_run_record,
    get_latest_completed_run_info,
    get_usable_feedback_session_ids,
)


MIN_NEW_FEEDBACK_SAMPLES = 5
MIN_TOTAL_TRAINING_SAMPLES = 10
MAX_DAYS_BETWEEN_RETRAINS = 7

LATEST_META_PATH = os.path.join(TRAINED_MODELS_DIR, "training_metadata.json")
RETRAIN_LOG_PATH = os.path.join(TRAINED_MODELS_DIR, "retrain_log.json")
RETRAIN_STATUS_PATH = os.path.join(TRAINED_MODELS_DIR, "retrain_status.json")


def write_status(status_dict):
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    with open(RETRAIN_STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2, ensure_ascii=False)


def get_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def load_latest_training_metadata():
    if not os.path.exists(LATEST_META_PATH):
        return None

    with open(LATEST_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_iso_datetime(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def get_last_training_time():
    metadata = load_latest_training_metadata()
    if not metadata:
        return None

    return parse_iso_datetime(metadata.get("trained_at"))


def fetch_feedback_session_stats(conn):
    query = """
        SELECT
            s.session_id,
            s.created_at
        FROM sessions s
        WHERE EXISTS (
            SELECT 1
            FROM tag_feedback tf
            WHERE tf.session_id = s.session_id
        )
        ORDER BY s.created_at ASC
    """
    cursor = conn.execute(query)
    rows = cursor.fetchall()

    return [
        {
            "session_id": row[0],
            "created_at": row[1],
        }
        for row in rows
    ]


def count_new_feedback_sessions(all_feedback_sessions, last_training_time):
    if last_training_time is None:
        return len(all_feedback_sessions)

    count = 0
    for row in all_feedback_sessions:
        created_at = parse_iso_datetime(row["created_at"])
        if created_at and created_at > last_training_time:
            count += 1

    return count


def should_retrain():
    latest_run = get_latest_completed_run_info()
    usable_session_ids = get_usable_feedback_session_ids()

    if latest_run is None:
        return {
            "should_retrain": len(usable_session_ids) >= 1,
            "reason": "No previous completed training run found.",
            "last_training_time": None,
            "days_since_last_training": None,
            "new_feedback_sessions": len(usable_session_ids),
            "total_feedback_sessions": len(usable_session_ids),
            "new_session_ids": usable_session_ids,
            "reuse_session_ids": [],
            "selected_session_ids": usable_session_ids,
            "policy": {
                "min_new_feedback_samples": MIN_NEW_FEEDBACK_SAMPLES,
                "min_total_training_samples": MIN_TOTAL_TRAINING_SAMPLES,
                "max_days_between_retrains": MAX_DAYS_BETWEEN_RETRAINS,
            }
        }

    last_training_time = latest_run.get("completed_at")
    last_used_session_ids = set(latest_run.get("session_ids", []))
    usable_session_ids_set = set(usable_session_ids)

    new_session_ids = sorted(list(usable_session_ids_set - last_used_session_ids))
    old_reusable_session_ids = sorted(list(usable_session_ids_set & last_used_session_ids))

    now = datetime.now()
    days_since_last_training = None
    time_condition_met = False

    if last_training_time is not None:
        delta = now - last_training_time
        days_since_last_training = delta.days
        time_condition_met = delta >= timedelta(days=MAX_DAYS_BETWEEN_RETRAINS)

    enough_new_data = len(new_session_ids) >= MIN_NEW_FEEDBACK_SAMPLES
    selected_session_ids = list(new_session_ids)
    reuse_session_ids = []

    if enough_new_data:
        return {
            "should_retrain": True,
            "reason": (
                f"Retraining triggered: new usable sessions "
                f"({len(new_session_ids)}) >= minimum required ({MIN_NEW_FEEDBACK_SAMPLES})."
            ),
            "last_training_time": last_training_time.isoformat() if last_training_time else None,
            "days_since_last_training": days_since_last_training,
            "new_feedback_sessions": len(new_session_ids),
            "total_feedback_sessions": len(usable_session_ids),
            "new_session_ids": new_session_ids,
            "reuse_session_ids": [],
            "selected_session_ids": selected_session_ids,
            "policy": {
                "min_new_feedback_samples": MIN_NEW_FEEDBACK_SAMPLES,
                "min_total_training_samples": MIN_TOTAL_TRAINING_SAMPLES,
                "max_days_between_retrains": MAX_DAYS_BETWEEN_RETRAINS,
            }
        }

    if time_condition_met:
        needed = max(0, MIN_TOTAL_TRAINING_SAMPLES - len(selected_session_ids))
        reuse_session_ids = old_reusable_session_ids[:needed]
        selected_session_ids.extend(reuse_session_ids)

        if len(selected_session_ids) >= 1:
            return {
                "should_retrain": True,
                "reason": (
                    "Retraining triggered: time condition met. "
                    f"Using {len(new_session_ids)} new sessions and "
                    f"{len(reuse_session_ids)} reused older sessions."
                ),
                "last_training_time": last_training_time.isoformat() if last_training_time else None,
                "days_since_last_training": days_since_last_training,
                "new_feedback_sessions": len(new_session_ids),
                "total_feedback_sessions": len(usable_session_ids),
                "new_session_ids": new_session_ids,
                "reuse_session_ids": reuse_session_ids,
                "selected_session_ids": selected_session_ids,
                "policy": {
                    "min_new_feedback_samples": MIN_NEW_FEEDBACK_SAMPLES,
                    "min_total_training_samples": MIN_TOTAL_TRAINING_SAMPLES,
                    "max_days_between_retrains": MAX_DAYS_BETWEEN_RETRAINS,
                }
            }

    return {
        "should_retrain": False,
        "reason": (
            "Retraining skipped: not enough new usable sessions and time condition not satisfied."
        ),
        "last_training_time": last_training_time.isoformat() if last_training_time else None,
        "days_since_last_training": days_since_last_training,
        "new_feedback_sessions": len(new_session_ids),
        "total_feedback_sessions": len(usable_session_ids),
        "new_session_ids": new_session_ids,
        "reuse_session_ids": [],
        "selected_session_ids": [],
        "policy": {
            "min_new_feedback_samples": MIN_NEW_FEEDBACK_SAMPLES,
            "min_total_training_samples": MIN_TOTAL_TRAINING_SAMPLES,
            "max_days_between_retrains": MAX_DAYS_BETWEEN_RETRAINS,
        }
    }


def save_retrain_log(summary):
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    with open(RETRAIN_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_retraining_pipeline(force=False):
    run_id = create_training_run_record(status="running", forced=force)

    try:
        decision_info = should_retrain()

        if not force and not decision_info["should_retrain"]:
            summary = {
                "status": "skipped",
                "timestamp": datetime.now().isoformat(),
                "forced": force,
                "run_id": run_id,
                **decision_info
            }
            save_retrain_log(summary)

            complete_training_run_record(
                run_id=run_id,
                status="skipped",
                model_version=None,
                session_ids=[],
                summary=summary,
                error_message=None,
            )
            return summary

        selected_session_ids = decision_info.get("selected_session_ids", [])
        dataset_result = build_dataset(selected_session_ids=selected_session_ids)
        training_metadata = train()

        model_version = training_metadata.get("version_id")
        used_session_ids = dataset_result.get("used_session_ids", [])

        summary = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "forced": force,
            "run_id": run_id,
            **decision_info,
            "dataset_result": {
                "x_shape": dataset_result.get("x_shape"),
                "y_shape": dataset_result.get("y_shape"),
                "stats": dataset_result.get("stats"),
                "used_session_ids_count": len(used_session_ids),
                "new_session_ids_count": len(decision_info.get("new_session_ids", [])),
                "reused_session_ids_count": len(decision_info.get("reuse_session_ids", [])),
            },
            "training_result": training_metadata
        }

        save_retrain_log(summary)

        complete_training_run_record(
            run_id=run_id,
            status="completed",
            model_version=model_version,
            session_ids=used_session_ids,
            summary=summary,
            error_message=None,
        )

        return summary

    except Exception as e:
        summary = {
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "forced": force,
            "run_id": run_id,
            "error": str(e),
        }
        save_retrain_log(summary)

        complete_training_run_record(
            run_id=run_id,
            status="failed",
            model_version=None,
            session_ids=[],
            summary=summary,
            error_message=str(e),
        )
        raise
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    write_status({
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "force": args.force
    })

    try:
        summary = run_retraining_pipeline(force=args.force)

        write_status({
            "status": summary["status"],
            "finished_at": datetime.now().isoformat(),
            "adaptive_model_reloaded": False,
            "summary": summary
        })
    except Exception as e:
        write_status({
            "status": "failed",
            "finished_at": datetime.now().isoformat(),
            "error": str(e)
        })
        raise


if __name__ == "__main__":
    main()