import os
import torch


# backend/ -> project root
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
TRAINING_DIR = os.path.join(DATA_DIR, "training")

# Database
DB_PATH = os.path.join(DATA_DIR, "app.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Model paths
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")

RAM_CHECKPOINT = os.path.join(PRETRAINED_DIR, "ram_plus_swin_large_14m.pth")
RAM_IMAGE_SIZE = 384
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# App behavior
MAX_TAGS = 8
STATUS_CHOICES = ["correct", "partially correct", "incorrect"]

# Adaptive model inference defaults
ADAPTIVE_THRESHOLD = 0.5
ADAPTIVE_TOP_K = 5