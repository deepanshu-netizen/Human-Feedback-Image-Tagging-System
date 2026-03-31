import os
import json
import time
import threading

from backend.config import TRAINED_MODELS_DIR
from backend.core.model_manager import model_manager
from backend.services.training_job_service import (
    is_training_running,
    start_retraining_job,
    get_retraining_status,
)
from training.retrain_pipeline import should_retrain


RETRAIN_STATUS_PATH = os.path.join(TRAINED_MODELS_DIR, "retrain_status.json")

# Project-scale setting: check once per minute
AUTOMATION_CHECK_INTERVAL_SECONDS = 60


class AutomationService:
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.started = False
        self.last_seen_completed_at = None

    def start(self):
        if self.started:
            return

        self.thread = threading.Thread(target=self.run_loop, daemon=True)
        self.thread.start()
        self.started = True
        print("[INFO] Automation service started.")

    def stop(self):
        self.stop_event.set()
        print("[INFO] Automation service stop requested.")

    def run_loop(self):
        while not self.stop_event.is_set():
            try:
                self.check_and_act()
            except Exception as e:
                print(f"[WARNING] Automation loop error: {e}")

            self.stop_event.wait(AUTOMATION_CHECK_INTERVAL_SECONDS)

    def check_and_act(self):
        status_data = get_retraining_status()
        status = status_data.get("status")

        # 1. If retraining just completed, auto-reload the adaptive model
        if status == "completed":
            finished_at = status_data.get("finished_at")
            already_reloaded = status_data.get("adaptive_model_reloaded", False)

            if finished_at != self.last_seen_completed_at and not already_reloaded:
                print("[INFO] Retraining completed. Reloading adaptive model...")
                model_manager.reload_adaptive_model()
                self.mark_model_reloaded(status_data)
                self.last_seen_completed_at = finished_at
                print("[INFO] Adaptive model auto-reloaded.")

            elif finished_at:
                self.last_seen_completed_at = finished_at

        # 2. If training is currently running, do nothing else
        if is_training_running():
            return

        # 3. If no training is running, check policy and auto-start if needed
        decision = should_retrain()
        if decision.get("should_retrain", False):
            print("[INFO] Retraining conditions satisfied. Starting retraining job...")
            result = start_retraining_job(force=False)
            print(f"[INFO] Retraining start result: {result}")

    def mark_model_reloaded(self, status_data):
        status_data["adaptive_model_reloaded"] = True
        status_data["adaptive_model_ready"] = model_manager.adaptive_ready

        os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
        with open(RETRAIN_STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)


automation_service = AutomationService()