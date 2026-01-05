import torch
import numpy as np

def extract_embedding(face_tensor, model):
    with torch.no_grad():
        emb = model(face_tensor)

    emb = emb.detach().cpu().numpy().flatten()
    emb = emb / np.linalg.norm(emb)
    return emb
