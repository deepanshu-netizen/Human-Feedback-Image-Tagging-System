import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

from backend.config import DEVICE, CLIP_MODEL_NAME
from shared.utils import normalize_image_input


class CLIPImageEncoder:
    def __init__(self):
        self.device = DEVICE
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.model.eval()

    def get_image_embedding(self, image):
        image = normalize_image_input(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_output = vision_outputs.pooler_output
            image_features = self.model.visual_projection(pooled_output)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = image_features[0].detach().cpu().numpy().astype(np.float32)
        return embedding