from fastapi import APIRouter, UploadFile, File, Depends
import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models_loader import yolo_model, facenet_model
from app.feature import extract_embedding
from app.crud import save_embeddings, load_all_embeddings
from app.recognition import recognize_face

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register")
async def register(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    embeddings = []

    for file in files:
        img = cv2.imdecode(
            np.frombuffer(await file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        results = yolo_model(img, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
        face = img[y1:y2, x1:x2]

        emb = extract_embedding(face, facenet_model)
        embeddings.append(emb)

    if not embeddings:
        return {"error": "No faces extracted"}

    person_id = save_embeddings(db, embeddings)
    return {"registered": True, "person_id": person_id, "images": len(embeddings)}


@router.post("/recognize")
async def recognize(file: UploadFile = File(...), db: Session = Depends(get_db)):
    img = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    results = yolo_model(img, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return {"recognized": False}

    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
    face = img[y1:y2, x1:x2]

    query_emb = extract_embedding(face, facenet_model)
    db_embs, db_ids = load_all_embeddings(db)

    pid, dist = recognize_face(query_emb, db_embs, db_ids)

    return {
        "recognized": pid is not None,
        "person_id": pid,
        "distance": dist
    }
