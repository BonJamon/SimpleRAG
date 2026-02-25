from langchain_openai import ChatOpenAI
from app.rag.templates import system_prompt_generator, get_user_prompt_generator
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from typing import AsyncIterator
import time

class Generator(ABC):
    @abstractmethod
    async def generate_answer(self, query:str, docs:List[Document])->AsyncIterator[str]:
        pass
        
class GeneratorV1(Generator):
    def __init__(self, logger, temperature=0.1, model_name="gpt-4o-mini", max_tokens=500):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
        self.logger = logger
        
    
    async def generate_answer(self, query, docs):
        start_time =time.time()
        messages = [("system", system_prompt_generator),
                    ("human", get_user_prompt_generator(query,docs))]
        
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content
            if chunk.usage_metadata:
                #Sends once at the end
                self.logger.info("llm_completed", 
                                 model_name=self.llm.model_name, 
                                 temperature=self.llm.temperature, 
                                 llm_latency_ms=1000*(time.time()-start_time), 
                                 input_tokens = chunk.usage_metadata["input_tokens"],
                                 output_tokens=chunk.usage_metadata["output_tokens"])
