import os
import uuid
import json
import numpy as np
from PIL import Image


def generate_session_id():
    return str(uuid.uuid4())


def normalize_image_input(image):
    """
    Safely convert Gradio image input into a clean RGB PIL image.
    Supports PIL.Image, numpy array, filepath string, or None.
    """
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, str):
        if os.path.exists(image):
            return Image.open(image).convert("RGB")
        raise ValueError(f"Image path does not exist: {image}")

    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")

    raise ValueError(f"Unsupported image input type: {type(image)}")


def dedupe_preserve_order(items):
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def normalize_new_tags(new_tags_text):
    if not new_tags_text:
        return []

    tags = [tag.strip().lower() for tag in new_tags_text.split(",")]
    tags = [tag for tag in tags if tag]
    return dedupe_preserve_order(tags)


def json_dumps_safe(value):
    return json.dumps(value, ensure_ascii=False)