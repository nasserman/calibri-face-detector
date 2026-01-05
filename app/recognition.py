import numpy as np
from svm_utils import load_svm

def recognize_face(embedding):
    svm = load_svm()

    proba = svm.predict_proba([embedding])[0]
    pred = svm.classes_[np.argmax(proba)]
    confidence = float(np.max(proba))

    return pred, confidence
