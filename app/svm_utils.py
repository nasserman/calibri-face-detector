import pickle
from sklearn.svm import SVC
from crud import load_all_embeddings

MODEL_PATH = "app/svm_model.pkl"

def train_svm(db):
    X, y = load_all_embeddings(db)
    if len(X) < 2:
        return None

    clf = SVC(
        kernel="linear",
        probability=True,
        C=1.0
    )
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    return clf

def load_svm():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
