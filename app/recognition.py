import numpy as np
from app.crud import get_all_embeddings


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def recognize_face(query_embedding: np.ndarray, threshold: float = 1.0):
    records = get_all_embeddings()

    best_distance = float("inf")
    best_student = None

    for record in records:
        db_embedding = record["embedding"]
        dist = euclidean_distance(query_embedding, db_embedding)

        if dist < best_distance:
            best_distance = dist
            best_student = record["student_id"]

    if best_distance <= threshold:
        return best_student, best_distance
from collections import defaultdict

def recognize_face(query_embedding, threshold=0.9):
    records = get_all_embeddings()

    grouped = defaultdict(list)
    for r in records:
        grouped[r["student_id"]].append(r["embedding"])

    best_dist = float("inf")
    best_student = None

    for student_id, embs in grouped.items():
        mean_emb = np.mean(embs, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        dist = np.linalg.norm(query_embedding - mean_emb)

        if dist < best_dist:
            best_dist = dist
            best_student = student_id

    if best_dist <= threshold:
        return best_student, best_dist

    return None, best_dist


