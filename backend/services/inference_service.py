import os
from typing import List

from backend.config import (
    MAX_TAGS,
    ADAPTIVE_THRESHOLD,
    ADAPTIVE_TOP_K,
    UPLOADS_DIR,
)
from backend.core.model_manager import model_manager
from shared.utils import generate_session_id, normalize_image_input


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        item = str(item).strip().lower()
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def merge_tags(ram_tags, adaptive_predictions, max_tags=MAX_TAGS):
    adaptive_tags = [item["tag"] for item in adaptive_predictions]
    merged = dedupe_preserve_order(list(ram_tags) + list(adaptive_tags))
    return merged[:max_tags]


def save_uploaded_image(image, session_id: str) -> str:
    image = normalize_image_input(image)
    image_path = os.path.join(UPLOADS_DIR, f"{session_id}.jpg")
    image.save(image_path)
    return image_path


def run_prediction(image):
    image = normalize_image_input(image)

    ram_tagger = model_manager.get_ram_tagger()
    ram_tags = ram_tagger.generate_tags(image)

    adaptive_predictions = []
    if model_manager.adaptive_ready:
        adaptive_predictor = model_manager.get_adaptive_predictor()
        adaptive_predictions = adaptive_predictor.predict_tags(
            image=image,
            threshold=ADAPTIVE_THRESHOLD,
            top_k=ADAPTIVE_TOP_K
        )

    combined_tags = merge_tags(ram_tags, adaptive_predictions, max_tags=MAX_TAGS)
    session_id = generate_session_id()

    return {
        "session_id": session_id,
        "ram_tags": ram_tags,
        "adaptive_tags": adaptive_predictions,
        "combined_tags": combined_tags,
    }