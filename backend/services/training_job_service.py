import os
import sys
import json
import subprocess

from backend.config import PROJECT_ROOT, TRAINED_MODELS_DIR


RETRAIN_STATUS_PATH = os.path.join(TRAINED_MODELS_DIR, "retrain_status.json")


def is_training_running():
    if not os.path.exists(RETRAIN_STATUS_PATH):
        return False

    try:
        with open(RETRAIN_STATUS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("status") == "running"
    except Exception:
        return False


def start_retraining_job(force: bool = False):
    if is_training_running():
        return {
            "started": False,
            "message": "Retraining is already running."
        }

    cmd = [sys.executable, "-m", "training.retrain_pipeline"]
    if force:
        cmd.append("--force")

    subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return {
        "started": True,
        "message": "Retraining process started.",
        "force": force
    }


def get_retraining_status():
    if not os.path.exists(RETRAIN_STATUS_PATH):
        return {
            "status": "not_started"
        }

    try:
        with open(RETRAIN_STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {
            "status": "unknown",
            "error": str(e)
        }