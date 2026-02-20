from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from app.rag.templates import system_prompt, get_user_prompt
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from typing import AsyncIterator

class Generator(ABC):
    @abstractmethod
    async def generate_answer(self, query:str, docs:List[Document])->AsyncIterator[str]:
        pass
        

load_dotenv()
class GeneratorV1(Generator):
    def __init__(self, temperature=0.1, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=300, streaming=True)
        
    
    async def generate_answer(self, query, docs):
        messages = [("system", system_prompt),
                    ("human", get_user_prompt(query,docs))]
        
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content
