import test.context
from app.rag.helper import catch_streaming_response
import asyncio
from app.services.rag import RAG


def test_answer_generation():
    query = "Hot to reset my router?"
    rag = RAG()
    a,_,_,_= asyncio.run(catch_streaming_response(rag.stream_answer,query=query))
    print(a)

    assert type(a)==str



