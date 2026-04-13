import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from backend.config import DATA_DIR, TRAINED_MODELS_DIR, DEVICE


TRAINING_DIR = os.path.join(DATA_DIR, "training")

X_PATH = os.path.join(TRAINING_DIR, "X_embeddings.npy")
Y_PATH = os.path.join(TRAINING_DIR, "Y_targets.npy")
Y_WEIGHTS_PATH = os.path.join(TRAINING_DIR, "Y_weights.npy")
VOCAB_PATH = os.path.join(TRAINING_DIR, "tag_vocab.json")

LATEST_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "adaptive_tagger.pt")
LATEST_META_PATH = os.path.join(TRAINED_MODELS_DIR, "training_metadata.json")

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
HIDDEN_DIM = 256
DROPOUT = 0.3


class EmbeddingTagDataset(Dataset):
    def __init__(self, x_path, y_path, y_weights_path):
        self.X = np.load(x_path).astype(np.float32)
        self.Y = np.load(y_path).astype(np.float32)
        self.Y_weights = np.load(y_weights_path).astype(np.float32)

        if len(self.X) != len(self.Y) or len(self.X) != len(self.Y_weights):
            raise ValueError(
                f"Mismatch lengths: X={len(self.X)}, Y={len(self.Y)}, Y_weights={len(self.Y_weights)}"
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        y_weights = torch.tensor(self.Y_weights[idx], dtype=torch.float32)
        return x, y, y_weights


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


def ensure_models_dir():
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)


def load_vocab():
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"tag_vocab.json not found: {VOCAB_PATH}")

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    if not isinstance(vocab, list) or len(vocab) == 0:
        raise ValueError("tag_vocab.json is empty or invalid.")

    return vocab


def generate_version_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_versioned_paths(version_id: str):
    model_path = os.path.join(TRAINED_MODELS_DIR, f"adaptive_tagger_{version_id}.pt")
    meta_path = os.path.join(TRAINED_MODELS_DIR, f"training_metadata_{version_id}.json")
    return model_path, meta_path


def compute_masked_bce_loss(logits, targets, weights):
    loss_matrix = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )
    weighted_loss = loss_matrix * weights
    denom = weights.sum().clamp_min(1e-8)
    return weighted_loss.sum() / denom


def train():
    ensure_models_dir()

    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X file not found: {X_PATH}")
    if not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Y file not found: {Y_PATH}")
    if not os.path.exists(Y_WEIGHTS_PATH):
        raise FileNotFoundError(f"Y weights file not found: {Y_WEIGHTS_PATH}")

    vocab = load_vocab()
    dataset = EmbeddingTagDataset(X_PATH, Y_PATH, Y_WEIGHTS_PATH)

    if len(dataset) == 0:
        raise ValueError("Training dataset is empty.")

    input_dim = dataset.X.shape[1]
    output_dim = dataset.Y.shape[1]

    dataloader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True
    )

    model = AdaptiveTagClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=output_dim,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    version_id = generate_version_id()
    versioned_model_path, versioned_meta_path = get_versioned_paths(version_id)

    print("Starting training...")
    print(f"Device: {DEVICE}")
    print(f"Samples: {len(dataset)}")
    print(f"Input dim: {input_dim}")
    print(f"Output dim: {output_dim}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Version: {version_id}")

    epoch_losses = []

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        for batch_x, batch_y, batch_weights in dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_weights = batch_weights.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = compute_masked_bce_loss(logits, batch_y, batch_weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {avg_loss:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": output_dim,
        "dropout": DROPOUT,
        "vocab_size": len(vocab),
        "version_id": version_id,
        "trained_at": datetime.now().isoformat(),
    }

    torch.save(checkpoint, versioned_model_path)
    torch.save(checkpoint, LATEST_MODEL_PATH)

    training_metadata = {
        "version_id": version_id,
        "trained_at": datetime.now().isoformat(),
        "device": DEVICE,
        "num_samples": len(dataset),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "batch_size": min(BATCH_SIZE, len(dataset)),
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "final_loss": epoch_losses[-1],
        "all_epoch_losses": epoch_losses,
        "versioned_model_path": versioned_model_path,
        "latest_model_path": LATEST_MODEL_PATH,
        "vocab_path": VOCAB_PATH,
        "training_targets": "soft targets + per-label weights"
    }

    with open(versioned_meta_path, "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2, ensure_ascii=False)

    with open(LATEST_META_PATH, "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2, ensure_ascii=False)

    print("\nTraining complete.")
    print(f"Versioned model saved to: {versioned_model_path}")
    print(f"Latest model updated at: {LATEST_MODEL_PATH}")
    print(f"Versioned metadata saved to: {versioned_meta_path}")
    print(f"Latest metadata updated at: {LATEST_META_PATH}")

    return training_metadata


if __name__ == "__main__":
    train()