from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
from app.services.rag import RAG
from app.rag.pipeline import Pipeline
from app.rag.retriever import RetrieverV1
from app.rag.generator import GeneratorV1
from app.models import Question
from app.core.logging import LogConfig
from structlog import BoundLogger
from structlog.contextvars import bind_contextvars, clear_contextvars
from dotenv import load_dotenv
import uuid
import time

load_dotenv()

logger = LogConfig().get_logger()
logger = logger.bind(app="chatbot")
app = FastAPI()

retriever = RetrieverV1(k=2)
generator = GeneratorV1(temperature=0.1, model_name="gpt-4o-mini", logger=logger)
pipeline = Pipeline(retriever, generator, logger=logger)
rag = RAG(pipeline)

origins = ["http://localhost"]
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )



@app.middleware("http")
async def logging_middleware(request, call_next):
    clear_contextvars()
    bind_contextvars(request_id=str(uuid.uuid4()))

    start = time.time()

    response = await call_next(request)

    if hasattr(response, "body_iterator"):
        original_iterator = response.body_iterator

        async def wrapped_iterator():
            async for chunk in original_iterator:
                yield chunk

            # stream finished
            latency = (time.time() - start) * 1000
            logger.info(
                "request_completed",
                status_code=response.status_code,
                total_latency_ms=latency,
            )

        response.body_iterator = wrapped_iterator()
        return response

    # Non-streaming case
    latency = (time.time() - start) * 1000
    logger.info(
        "request_completed",
        status_code=response.status_code,
        total_latency_ms=latency,
    )

    return response


def get_rag() -> RAG:
    return rag

def get_logger() -> BoundLogger:
    return logger


@app.get("/")
async def main():
    return {"message": "Hello World"}


@app.post("/question")
async def generate_answer(question: Question, rag: RAG = Depends(get_rag), logger = Depends(get_logger)):
    logger.info("request received", question_length=len(question.question))
    return StreamingResponse(rag.stream_answer(question.question), media_type="text/event-stream")

