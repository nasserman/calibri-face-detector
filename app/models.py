from sqlalchemy import Column, Integer, String, LargeBinary
from app.database import Base


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
