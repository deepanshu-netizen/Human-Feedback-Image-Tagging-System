import os
import torch
from ram.models import ram_plus
from ram import inference_ram as ram_inference
from ram import get_transform

from backend.config import DEVICE, RAM_CHECKPOINT, RAM_IMAGE_SIZE
from shared.utils import normalize_image_input, dedupe_preserve_order


class RAMTagger:
    def __init__(self):
        if not os.path.exists(RAM_CHECKPOINT):
            raise FileNotFoundError(
                f"RAM++ checkpoint not found at: {RAM_CHECKPOINT}"
            )

        self.device = DEVICE
        self.transform = get_transform(image_size=RAM_IMAGE_SIZE)

        self.model = ram_plus(
            pretrained=RAM_CHECKPOINT,
            image_size=RAM_IMAGE_SIZE,
            vit="swin_l"
        )
        self.model.eval()
        self.model = self.model.to(self.device)

    def generate_tags(self, image):
        """
        Input:
            image -> PIL image / numpy array / file path
        Output:
            list[str]
        """
        image = normalize_image_input(image)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = ram_inference(image_tensor, self.model)

        english_tags = result[0]
        raw_tags = [tag.strip().lower() for tag in english_tags.split("|") if tag.strip()]
        tags = dedupe_preserve_order(raw_tags)
        return tags