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


RAM_BASE_SCORE = 0.55
RAM_AND_ADAPTIVE_BONUS_MULTIPLIER = 0.35
MIN_ADAPTIVE_ONLY_SCORE = 0.35


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
    ordered_tags = dedupe_preserve_order(
        list(ram_tags) + [item["tag"] for item in adaptive_predictions]
    )
    first_seen_index = {tag: idx for idx, tag in enumerate(ordered_tags)}

    adaptive_score_map = {}
    for item in adaptive_predictions:
        tag = str(item["tag"]).strip().lower()
        score = float(item.get("score", 0.0))
        adaptive_score_map[tag] = max(score, adaptive_score_map.get(tag, 0.0))

    combined_scores = {}

    for tag in ram_tags:
        tag = str(tag).strip().lower()
        if not tag:
            continue

        combined_scores[tag] = max(combined_scores.get(tag, 0.0), RAM_BASE_SCORE)

        if tag in adaptive_score_map:
            combined_scores[tag] = max(
                combined_scores[tag],
                RAM_BASE_SCORE + RAM_AND_ADAPTIVE_BONUS_MULTIPLIER * adaptive_score_map[tag]
            )

    for tag, score in adaptive_score_map.items():
        if tag not in combined_scores and score >= MIN_ADAPTIVE_ONLY_SCORE:
            combined_scores[tag] = score

    ranked = sorted(
        combined_scores.items(),
        key=lambda item: (-item[1], first_seen_index.get(item[0], 10**9))
    )

    return [tag for tag, _ in ranked[:max_tags]]


def save_uploaded_image(image, session_id: str) -> str:
    image = normalize_image_input(image)
    image_path = os.path.join(UPLOADS_DIR, f"{session_id}.jpg")
    image.save(image_path)
    return image_path


def run_prediction(image, max_tags=MAX_TAGS):
    image = normalize_image_input(image)
    max_tags = max(1, int(max_tags))

    ram_tagger = model_manager.get_ram_tagger()
    ram_tags = ram_tagger.generate_tags(image)

    adaptive_predictions = []
    if model_manager.adaptive_ready:
        adaptive_predictor = model_manager.get_adaptive_predictor()
        adaptive_predictions = adaptive_predictor.predict_tags(
            image=image,
            threshold=ADAPTIVE_THRESHOLD,
            top_k=max(max_tags, ADAPTIVE_TOP_K)
        )

    combined_tags = merge_tags(
        ram_tags,
        adaptive_predictions,
        max_tags=max_tags
    )
    session_id = generate_session_id()

    return {
        "session_id": session_id,
        "ram_tags": ram_tags,
        "adaptive_tags": adaptive_predictions,
        "combined_tags": combined_tags,
    }