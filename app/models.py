from sqlalchemy import Column, Integer, LargeBinary, String
from app.database import Base
import pickle


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True)
    person_id = Column(String, unique=True, index=True)
    embedding = Column(LargeBinary)

    def set_embedding(self, emb):
        self.embedding = pickle.dumps(emb)

    def get_embedding(self):
        return pickle.loads(self.embedding)
