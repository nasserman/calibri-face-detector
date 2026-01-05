from fastapi import FastAPI
from app.api import router as recognize_router
from app.embedding_api import router as register_router
from app.database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(register_router, tags=["Register"])
app.include_router(recognize_router, tags=["Recognize"])
