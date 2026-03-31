import os
import json
import sqlite3
import numpy as np

from backend.config import DB_PATH, DATA_DIR


TRAINING_DIR = os.path.join(DATA_DIR, "training")
X_PATH = os.path.join(TRAINING_DIR, "X_embeddings.npy")
Y_PATH = os.path.join(TRAINING_DIR, "Y_labels.npy")
VOCAB_PATH = os.path.join(TRAINING_DIR, "tag_vocab.json")
META_PATH = os.path.join(TRAINING_DIR, "dataset_metadata.json")


def ensure_training_dir():
    os.makedirs(TRAINING_DIR, exist_ok=True)


def get_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def fetch_sessions(conn):
    query = """
        SELECT
            session_id,
            image_path,
            embedding_path,
            extracted_tags,
            new_tags_added,
            num_generated_tags,
            num_new_tags,
            created_at
        FROM sessions
    """
    cursor = conn.execute(query)
    columns = [col[0] for col in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return rows


def fetch_tag_feedback(conn):
    query = """
        SELECT
            session_id,
            image_path,
            embedding_path,
            generated_tag,
            human_status,
            created_at
        FROM tag_feedback
    """
    cursor = conn.execute(query)
    columns = [col[0] for col in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return rows


def safe_parse_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip().lower() for x in value if str(x).strip()]

    value = str(value).strip()
    if not value:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except Exception:
        pass

    parts = [x.strip().lower() for x in value.split(",")]
    return [x for x in parts if x]


def dedupe_preserve_order(items):
    seen = set()
    output = []
    for item in items:
        item = str(item).strip().lower()
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def normalize_status(status):
    if status is None:
        return ""

    status = str(status).strip().lower()

    mapping = {
        "correct": "correct",
        "partially correct": "partial",
        "partial": "partial",
        "incorrect": "incorrect",
    }
    return mapping.get(status, status)


def build_feedback_index(tag_feedback_rows):
    feedback_by_session = {}

    for row in tag_feedback_rows:
        session_id = str(row["session_id"])
        feedback_by_session.setdefault(session_id, []).append(row)

    return feedback_by_session


def get_positive_tags_for_session(session_row, feedback_rows_for_session):
    """
    Label rules:
    - correct -> positive
    - new tags added -> positive
    - incorrect -> ignored
    - partial -> ignored
    """
    positive_tags = []

    for row in feedback_rows_for_session:
        tag = str(row.get("generated_tag", "")).strip().lower()
        status = normalize_status(row.get("human_status", ""))

        if not tag:
            continue

        if status == "correct":
            positive_tags.append(tag)

    new_tags = safe_parse_list(session_row.get("new_tags_added", ""))
    positive_tags.extend(new_tags)

    return dedupe_preserve_order(positive_tags)


def load_embedding(embedding_path):
    if embedding_path is None:
        return None

    embedding_path = str(embedding_path).strip()
    if not embedding_path:
        return None

    if not os.path.exists(embedding_path):
        print(f"[WARNING] Embedding file not found: {embedding_path}")
        return None

    try:
        embedding = np.load(embedding_path)
    except Exception as e:
        print(f"[WARNING] Could not load embedding {embedding_path}: {e}")
        return None

    if embedding.ndim != 1:
        embedding = embedding.reshape(-1)

    return embedding.astype(np.float32)


def build_tag_vocab(session_rows, feedback_by_session):
    vocab_tags = set()

    for session_row in session_rows:
        session_id = str(session_row["session_id"])
        feedback_rows = feedback_by_session.get(session_id, [])
        positive_tags = get_positive_tags_for_session(session_row, feedback_rows)

        for tag in positive_tags:
            vocab_tags.add(tag)

    vocab_list = sorted(vocab_tags)
    tag_to_index = {tag: idx for idx, tag in enumerate(vocab_list)}
    return vocab_list, tag_to_index


def build_dataset(selected_session_ids=None):
    ensure_training_dir()

    conn = get_connection()
    try:
        session_rows = fetch_sessions(conn)
        tag_feedback_rows = fetch_tag_feedback(conn)
    finally:
        conn.close()

    if selected_session_ids is not None:
        selected_session_ids = set(str(x) for x in selected_session_ids)
        session_rows = [
            row for row in session_rows
            if str(row["session_id"]) in selected_session_ids
        ]

    if len(session_rows) == 0:
        raise ValueError("No matching session records found for dataset building.")

    feedback_by_session = build_feedback_index(tag_feedback_rows)
    vocab_list, tag_to_index = build_tag_vocab(session_rows, feedback_by_session)

    if len(vocab_list) == 0:
        raise ValueError("No positive tags found. Nothing to build yet.")

    X = []
    Y = []
    used_session_ids = []

    stats = {
        "sessions_total": 0,
        "sessions_used": 0,
        "sessions_skipped_missing_embedding": 0,
        "sessions_skipped_no_positive_tags": 0,
        "embedding_dim": None,
        "num_tags_in_vocab": len(vocab_list),
        "data_source": "sqlite",
        "db_path": DB_PATH,
    }

    for session_row in session_rows:
        stats["sessions_total"] += 1

        session_id = str(session_row["session_id"])
        embedding_path = session_row.get("embedding_path", "")

        embedding = load_embedding(embedding_path)
        if embedding is None:
            stats["sessions_skipped_missing_embedding"] += 1
            continue

        feedback_rows = feedback_by_session.get(session_id, [])
        positive_tags = get_positive_tags_for_session(session_row, feedback_rows)

        if len(positive_tags) == 0:
            stats["sessions_skipped_no_positive_tags"] += 1
            continue

        if stats["embedding_dim"] is None:
            stats["embedding_dim"] = int(embedding.shape[0])

        y = np.zeros(len(vocab_list), dtype=np.float32)
        for tag in positive_tags:
            if tag in tag_to_index:
                y[tag_to_index[tag]] = 1.0

        X.append(embedding)
        Y.append(y)
        stats["sessions_used"] += 1
        used_session_ids.append(session_id)

    if len(X) == 0:
        raise ValueError("No usable sessions found after filtering.")

    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)

    np.save(X_PATH, X)
    np.save(Y_PATH, Y)

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, indent=2, ensure_ascii=False)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "x_shape": list(X.shape),
                "y_shape": list(Y.shape),
                **stats,
                "label_rules": {
                    "correct": "positive",
                    "new_tags_added": "positive",
                    "incorrect": "ignored",
                    "partial": "ignored"
                }
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Dataset built successfully.")
    print(f"Saved X: {X_PATH} | shape={X.shape}")
    print(f"Saved Y: {Y_PATH} | shape={Y.shape}")
    print(f"Saved vocab: {VOCAB_PATH} | num_tags={len(vocab_list)}")
    print(f"Saved metadata: {META_PATH}")
    print("\nSummary:")
    for k, v in stats.items():
        print(f"- {k}: {v}")
        
    return {
        "x_shape": list(X.shape),
        "y_shape": list(Y.shape),
        "stats": stats,
        "used_session_ids": used_session_ids,
        "vocab_list": vocab_list,
    }


if __name__ == "__main__":
    build_dataset()