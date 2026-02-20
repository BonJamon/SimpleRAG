from app.rag.retriever import Retriever
from app.rag.generator import Generator
from typing import AsyncIterator
class Pipeline:
    def __init__(self,retriever: Retriever, generator:Generator):
        self.retriever = retriever
        self.generator = generator
    
    async def stream_answer(self, query:str)->AsyncIterator[str]:
        #Retrieve relevant top-k documents
        docs = await self.retriever.retrieve_documents(query)
        #Ignore metadata
        docs_clean = [doc.page_content for doc in docs]
        #Stream generated answer
        async for chunk in self.generator.generate_answer(query, docs_clean):
            yield chunk