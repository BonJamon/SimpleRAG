from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
from app.services.rag import RAG
from app.rag.pipeline import Pipeline
from app.rag.retriever import RetrieverV1
from app.rag.generator import GeneratorV1
from app.models import Question


app = FastAPI()

retriever = RetrieverV1(k=2)
generator = GeneratorV1()
pipeline = Pipeline(retriever, generator)
rag = RAG(pipeline)

origins = ["http://localhost"]
app.add_middleware(CORSMiddleware,
                   allow_origins=[origins],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )


def get_rag() -> RAG:
    return rag



@app.get("/")
async def main():
    return {"message": "Hello World"}


@app.post("/question")
async def generate_answer(question: Question, rag: RAG = Depends(get_rag)):
    return StreamingResponse(rag.stream_answer(question.question), media_type="text/event-stream")

