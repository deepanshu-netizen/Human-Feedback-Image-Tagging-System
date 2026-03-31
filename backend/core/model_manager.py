from backend.core.ram_tagger import RAMTagger
from backend.core.clip_encoder import CLIPImageEncoder
from backend.core.adaptive_predictor import AdaptivePredictor


class ModelManager:
    def __init__(self):
        self.ram_tagger = None
        self.clip_encoder = None
        self.adaptive_predictor = None
        self.adaptive_ready = False

    def load_all_models(self):
        self.ram_tagger = RAMTagger()
        self.clip_encoder = CLIPImageEncoder()
        self.reload_adaptive_model()

    def reload_adaptive_model(self):
        try:
            if self.clip_encoder is None:
                raise RuntimeError("CLIPImageEncoder must be loaded before AdaptivePredictor.")

            self.adaptive_predictor = AdaptivePredictor(
                clip_encoder=self.clip_encoder
            )
            self.adaptive_ready = True
            print("[INFO] Adaptive predictor loaded successfully.")
        except Exception as e:
            self.adaptive_predictor = None
            self.adaptive_ready = False
            print(f"[INFO] Adaptive predictor not available: {e}")

    def get_ram_tagger(self):
        if self.ram_tagger is None:
            raise RuntimeError("RAMTagger is not loaded.")
        return self.ram_tagger

    def get_clip_encoder(self):
        if self.clip_encoder is None:
            raise RuntimeError("CLIPImageEncoder is not loaded.")
        return self.clip_encoder

    def get_adaptive_predictor(self):
        return self.adaptive_predictor


model_manager = ModelManager()