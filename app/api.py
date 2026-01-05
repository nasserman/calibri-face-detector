from fastapi import APIRouter, UploadFile, File
import cv2, torch
import numpy as np

from models_loader import yolo_model, facenet_model
from feature import extract_embedding
from recognition import recognize_face

router = APIRouter()

@router.post("/face/recognize")
async def recognize(file: UploadFile = File(...)):
    img = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    results = yolo_model(img, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return {"recognized": False}

    box = boxes[0]
    x1,y1,x2,y2 = map(int, box.xyxy[0])
    face = img[y1:y2, x1:x2]

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160,160))

    tensor = torch.tensor(face).permute(2,0,1).float().unsqueeze(0)/255
    tensor = tensor.to(facenet_model.device)

    emb = extract_embedding(tensor, facenet_model)

    label, conf = recognize_face(emb)

    return {
        "recognized": True,
        "identity": label,
        "confidence": conf
    }
