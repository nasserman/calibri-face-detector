import numpy as np


def recognize_face(query_emb, db_embs, db_ids, threshold=0.9):
    if len(db_embs) == 0:
        return None, None

    dists = np.linalg.norm(db_embs - query_emb, axis=1)
    idx = np.argmin(dists)

    if dists[idx] < threshold:
        return db_ids[idx], float(dists[idx])

    return None, float(dists[idx])
