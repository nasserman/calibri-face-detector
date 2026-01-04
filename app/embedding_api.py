from fastapi import APIRouter, UploadFile, File, Depends
import cv2
import numpy as np
import torch

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models_loader import yolo_model, facenet_model
from app.feature import extract_embedding
from app.crud import save_embedding

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register")
async def register_face(
    student_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    results = yolo_model(img, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return {"error": "No face detected"}

    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
    face = img[y1:y2, x1:x2]

    embedding = extract_embedding(face, facenet_model)
    save_embedding(db, student_id, embedding)

    return {"status": "registered", "student_id": student_id}
