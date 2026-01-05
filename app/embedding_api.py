from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
import cv2, torch
import numpy as np

from app.models_loader import device
from app.models_loader import yolo_model, facenet_model
from app.feature import extract_embedding
from app.crud import add_embedding
from app.svm_utils import train_svm
from app.database import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/face/register")
async def register_faces(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    for file in files:
        label = file.filename.split(".")[0]  #  آیدی = اسم فایل

        img = cv2.imdecode(
            np.frombuffer(await file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        results = yolo_model(img, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (160, 160))

            tensor = (
            torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device))


            emb = extract_embedding(tensor, facenet_model)
            add_embedding(db, label, emb)

    train_svm(db)

    return {"status": "registered"}
