from sqlalchemy.orm import Session
from app.models import FaceEmbedding
import numpy as np

def save_embeddings(db: Session, person_name: str, embedding: list):
    """
    ذخیره سازی امبدینگ به همراه نام استخراج شده از فایل
    """
    # ایجاد یک رکورد جدید در دیتابیس
    # توجه: در مدل FaceEmbedding، ستون person_id باید از نوع String باشد
    rec = FaceEmbedding(person_id=person_name)
    rec.set_embedding(embedding)
    
    db.add(rec)
    db.commit()
    return person_name


def load_all_embeddings(db: Session):
    """
    دریافت تمام امبدینگ‌ها و شناسه‌های متنی از دیتابیس
    """
    records = db.query(FaceEmbedding).all()

    embs, ids = [], []
    for r in records:
        emb = r.get_embedding()
        if emb is not None:
            embs.append(emb)
            ids.append(r.person_id) # اینجا person_id همان نام فایل (String) است

    return np.array(embs), ids