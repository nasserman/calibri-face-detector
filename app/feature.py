import cv2
import numpy as np
import torch


def extract_embedding(face_bgr, facenet_model):
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))

    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    tensor = torch.from_numpy(face)
    device = next(facenet_model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        emb = facenet_model(tensor)

    emb = emb.cpu().numpy().flatten()
    emb = emb / np.linalg.norm(emb)
    return emb
