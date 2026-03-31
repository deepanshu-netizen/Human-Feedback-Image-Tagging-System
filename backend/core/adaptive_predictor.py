import os
import json
import torch
import torch.nn as nn

from backend.config import PROJECT_ROOT
from shared.utils import normalize_image_input


TRAINING_DIR = os.path.join(PROJECT_ROOT, "data", "training")
MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")

VOCAB_PATH = os.path.join(TRAINING_DIR, "tag_vocab.json")
MODEL_PATH = os.path.join(MODELS_DIR, "adaptive_tagger.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_THRESHOLD = 0.5
TOP_K = 10


class AdaptiveTagClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class AdaptivePredictor:
    def __init__(
        self,
        clip_encoder,
        model_path=MODEL_PATH,
        vocab_path=VOCAB_PATH,
        device=DEVICE
    ):
        self.device = device
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.clip_encoder = clip_encoder

        self.vocab = self._load_vocab()
        self.model = self._load_model()

    def _load_vocab(self):
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        if not isinstance(vocab, list) or len(vocab) == 0:
            raise ValueError("Vocabulary file is empty or invalid.")

        return vocab

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        model = AdaptiveTagClassifier(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"],
            dropout=checkpoint["dropout"]
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def predict_scores(self, image):
        image = normalize_image_input(image)
        embedding = self.clip_encoder.get_image_embedding(image)

        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        results = []
        for tag, score in zip(self.vocab, probs):
            results.append({
                "tag": tag,
                "score": float(score)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def predict_tags(self, image, threshold=DEFAULT_THRESHOLD, top_k=TOP_K):
        scored_results = self.predict_scores(image)

        selected = [item for item in scored_results if item["score"] >= threshold]

        if len(selected) == 0:
            selected = scored_results[:top_k]
        else:
            selected = selected[:top_k]

        return selected