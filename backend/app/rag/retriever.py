
from langchain_community.vectorstores import FAISS
from app.rag.embeddings import E5Embeddings
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class Retriever(ABC):
    @abstractmethod
    async def retrieve_documents(self, query:str)->List[Document]:
        pass


class RetrieverV1(Retriever):
    def __init__(self,k, vectorstore_path="data/vectorstore/faiss_index"):
        self.retriever = self.get_retriever(vectorstore_path)
        self.k = k

    async def retrieve_documents(self, query):
        contexts = await self.retriever.ainvoke(query)
        return contexts[:self.k]
    
    @staticmethod
    def get_retriever(vectorstore_path):
        embeddings = E5Embeddings()
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        return retriever
