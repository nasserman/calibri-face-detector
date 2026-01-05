import numpy as np
from sqlalchemy.orm import Session
from models import FaceEmbedding

def add_embedding(db: Session, label: str, embedding: np.ndarray):
    record = FaceEmbedding(
        label=label,
        embedding=embedding.astype(np.float32).tobytes()
    )
    db.add(record)
    db.commit()

def load_all_embeddings(db: Session):
    records = db.query(FaceEmbedding).all()
    X, y = [], []

    for r in records:
        X.append(np.frombuffer(r.embedding, dtype=np.float32))
        y.append(r.label)

    return np.array(X), np.array(y)
