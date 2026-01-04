from fastapi import APIRouter, UploadFile, File, Depends
import cv2
import numpy as np
from sqlalchemy.orm import Session
import os

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
    registration_results = []

    for file in files:
        # ۱. استخراج نام از فایل (حذف پسوند)
        # مثال: "Ali_Reza.jpg" -> "Ali_Reza"
        filename_raw = os.path.splitext(file.filename)[0]
        
        # خواندن تصویر
        content = await file.read()
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            continue

        # تشخیص چهره با YOLO
        results = yolo_model(img, verbose=False)
        boxes = results[0].boxes
        
        if boxes is None or len(boxes) == 0:
            continue

        # استخراج اولین چهره یافت شده
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
        face = img[y1:y2, x1:x2]

        # استخراج ویژگی (Embedding)
        emb = extract_embedding(face, facenet_model)
        
        # ۲. ذخیره در دیتابیس همراه با نام استخراج شده از فایل
        # توجه: تابع save_embeddings باید دو آرگومان (db, data) یا مشابه آن را بپذیرد
        save_embeddings(db, person_name=filename_raw, embedding=emb)
        
        registration_results.append(filename_raw)

    if not registration_results:
        return {"error": "No faces extracted from the provided images"}

    return {
        "registered": True, 
        "registered_names": list(set(registration_results)), 
        "total_images": len(registration_results)
    }

@router.post("/recognize")
async def recognize(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    results = yolo_model(img, verbose=False)
    boxes = results[0].boxes
    
    if boxes is None or len(boxes) == 0:
        return {"recognized": False, "message": "No face detected"}

    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
    face = img[y1:y2, x1:x2]

    query_emb = extract_embedding(face, facenet_model)
    
    # ۳. لود کردن امبدینگ‌ها و نام‌ها (IDs) از دیتابیس
    db_embs, db_names = load_all_embeddings(db)

    # تشخیص هویت
    name, dist = recognize_face(query_emb, db_embs, db_names)

    return {
        "recognized": name is not None,
        "person_name": name, # اینجا نام فایل قبلی نمایش داده می‌شود
        "distance": float(dist) if dist is not None else None
    }