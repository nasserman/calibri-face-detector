import numpy as np
from app.database import SessionLocal
from app.models import FaceEmbedding


def save_embedding(student_id: str, embedding: np.ndarray):
    db = SessionLocal()

    emb_bytes = embedding.astype(np.float32).tobytes()

    record = FaceEmbedding(
        student_id=student_id,
        embedding=emb_bytes
    )

    db.add(record)
    db.commit()
    db.close()


def get_all_embeddings():
    db = SessionLocal()
    records = db.query(FaceEmbedding).all()

    results = []
    for r in records:
        emb = np.frombuffer(r.embedding, dtype=np.float32)
        results.append({
            "student_id": r.student_id,
            "embedding": emb
        })

    db.close()
    return results
