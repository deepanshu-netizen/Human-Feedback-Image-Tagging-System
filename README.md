# Human-in-the-Loop Adaptive Image Tagging System

A full-stack, continuously learning image tagging system that combines pretrained vision models, user feedback, and automated retraining to improve tag quality over time.

## Overview

This project is a human-in-the-loop adaptive image tagging system designed to bridge the gap between powerful pretrained models and user-specific tagging requirements.

Instead of training an image tagging model from scratch, the system uses:

- **RAM++** for strong initial tag generation
- **CLIP** for semantic image embeddings
- **A custom adaptive multi-label classifier** for learning from feedback
- **An automated retraining pipeline** for continuous improvement

The system evolves through the following loop:

**Image → Initial Tags → User Feedback → Training Data → Model Update → Better Future Tags**

---

## Key Features

- Pretrained image tagging using **RAM++**
- Semantic image representation using **CLIP**
- Adaptive multi-label classifier trained on user feedback
- Human-in-the-loop feedback collection through a web interface
- Support for:
  - correct tags
  - partially correct tags
  - incorrect tags
  - user-added new tags
- Automated retraining and model reload
- Tag ranking and merging strategy for final predictions
- Database-backed session, feedback, and training traceability
- Cloud deployment support

---

## System Architecture

The system is organized into the following layers:

### 1. Frontend
Allows users to:
- upload an image
- view predicted tags
- mark tags as correct / partially correct / incorrect
- add missing tags
- submit feedback

### 2. API Layer
Built with **FastAPI**, exposing endpoints for:
- `/predict`
- `/submit-feedback`
- `/admin/retrain/start`
- `/admin/retrain/status`
- `/admin/retrain/reload-model`

### 3. Service Layer
Contains the core business logic:
- `inference_service`
- `feedback_service`
- `training_job_service`
- `automation_service`

### 4. ML Core
Includes:
- **RAM++** pretrained tagger
- **CLIP** image encoder
- **Adaptive classifier** for personalized learning

### 5. Data Layer
Tracks:
- image sessions
- tag feedback
- training runs
- training-session associations

### 6. Training Pipeline
Handles:
- dataset construction
- model training
- retraining decisions
- model versioning

---

## Machine Learning Design

### Base Tagger
The system uses **RAM++** to generate strong initial tags and solve the cold-start problem.

### Feature Extractor
**CLIP** converts images into semantic embedding vectors that are used for both inference and retraining.

### Adaptive Model
A lightweight feedforward neural network learns from feedback and predicts tag probabilities for known tags.

### Feedback-Aware Training
The training pipeline uses different supervision signals:

- **correct** → positive supervision
- **incorrect** → negative supervision
- **partial** → soft supervision
- **new tags added by user** → positive supervision

This makes the model more responsive to real user preferences over time.

---

## Tech Stack

### Backend
- Python
- FastAPI
- SQLAlchemy
- SQLite

### Machine Learning
- PyTorch
- Hugging Face Transformers
- CLIP
- RAM++

### Frontend
- HTML
- CSS
- JavaScript

### Storage
- SQLite database
- NumPy embeddings
- Trained model checkpoints

### Deployment
- AWS EC2

---

## Project Structure

```bash
Human-Feedback-Image-Tagging-System/
│
├── backend/
│   ├── api/
│   │   ├── predict.py
│   │   ├── feedback.py
│   │   └── training.py
│   │
│   ├── core/
│   │   ├── ram_tagger.py
│   │   ├── clip_encoder.py
│   │   ├── adaptive_predictor.py
│   │   └── model_manager.py
│   │
│   ├── db/
│   │   ├── database.py
│   │   ├── models.py
│   │   └── crud.py
│   │
│   ├── schemas/
│   │   ├── predict_schema.py
│   │   └── feedback_schema.py
│   │
│   ├── services/
│   │   ├── inference_service.py
│   │   ├── feedback_service.py
│   │   ├── training_job_service.py
│   │   └── automation_service.py
│   │
│   ├── config.py
│   └── main.py
│
├── training/
│   ├── build_dataset.py
│   ├── train_classifier.py
│   ├── retrain_pipeline.py
│   └── training_db.py
│
├── shared/
│   └── utils.py
│
├── data/
│   ├── uploads/
│   ├── embeddings/
│   └── training/
│
├── trained_models/
├── pretrained/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── README.md
