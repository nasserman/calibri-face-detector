from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import os

from app.database import engine
from app.models import Base
from app.feature import extract_embedding
from app.crud import save_embedding
from app.recognition import recognize_face

app = FastAPI()

# ساخت جدول‌ها
Base.metadata.create_all(bind=engine)


def extract_student_id(filename: str) -> str:
    return os.path.basename(filename).split("_")[0]


@app.post("/register/")
async def register_student(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        student_id = extract_student_id(file.filename)

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        embedding = extract_embedding(image)
        save_embedding(student_id, embedding)

        results.append({
            "filename": file.filename,
            "student_id": student_id
        })

    return {
        "message": "Registered successfully",
        "count": len(results),
        "details": results
    }


@app.post("/recognize/")
async def recognize_student(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    embedding = extract_embedding(image)
    student_id, score = recognize_face(embedding)

    if student_id is None:
        return {
            "recognized": False,
            "confidence": float(score)
        }

    return {
        "recognized": True,
        "student_id": student_id,
        "confidence": float(score)
    }
