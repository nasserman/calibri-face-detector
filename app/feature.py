import torch
import numpy as np

def extract_embedding(face_tensor, model):
    with torch.no_grad():
        emb = model(face_tensor)

    emb = emb.cpu().numpy().flatten()
    emb = emb / np.linalg.norm(emb)  # مهم برای SVM
    return emb
