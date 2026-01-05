from sqlalchemy import Column, Integer, String, LargeBinary
from app.database import Base

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)  # filename
    embedding = Column(LargeBinary)
