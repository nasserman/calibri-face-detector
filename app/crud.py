from sqlalchemy.orm import Session
from app.models import FaceEmbedding
import numpy as np


def get_next_person_id(db: Session):
    last = db.query(FaceEmbedding).order_by(FaceEmbedding.person_id.desc()).first()
    return 0 if last is None else last.person_id + 1


def save_embeddings(db: Session, embeddings: list):
    person_id = get_next_person_id(db)

    for emb in embeddings:
        rec = FaceEmbedding(person_id=person_id)
        rec.set_embedding(emb)
        db.add(rec)

    db.commit()
    return person_id


def load_all_embeddings(db: Session):
    records = db.query(FaceEmbedding).all()

    embs, ids = [], []
    for r in records:
        embs.append(r.get_embedding())
        ids.append(r.person_id)

    return np.array(embs), ids
