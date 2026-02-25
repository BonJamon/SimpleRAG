from app.rag.retriever import Retriever
from app.rag.generator import Generator
from typing import AsyncIterator
from structlog import BoundLogger
import time
class Pipeline:
    def __init__(self,retriever: Retriever, generator:Generator, logger:BoundLogger):
        self.retriever = retriever
        self.generator = generator
        self.logger = logger
    
    async def stream_answer(self, standalone_question:str, original_question:str)->AsyncIterator[str]:
        #Retrieve relevant top-k documents
        start_time = time.time()
        docs = await self.retriever.retrieve_documents(standalone_question)
        retrieval_time = time.time() - start_time
        retrieved_doc_ids = [doc.id for doc in docs]
        self.logger.info("retrieval finished", retrieval_latency_ms=1000*retrieval_time, retrieved_doc_ids=retrieved_doc_ids)
        #Ignore metadata
        docs_clean = [doc.page_content for doc in docs]
        #Stream generated answer
        async for chunk in self.generator.generate_answer(original_question, docs_clean):
            yield chunk