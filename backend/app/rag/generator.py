from langchain_openai import ChatOpenAI
from nemoguardrails import LLMRails
from app.rag.templates import system_prompt_generator, get_user_prompt_generator
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from typing import AsyncIterator
import tiktoken
import time

class Generator(ABC):
    @abstractmethod
    async def generate_answer(self, query:str, docs:List[Document])->AsyncIterator[str]:
        pass
        
class GeneratorV1(Generator):
    def __init__(self, logger, rails_config, temperature=0.1, model_name="gpt-4o-mini", max_tokens=500):
        llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens, streaming=True)
        self.model_name = model_name
        self.temperature = temperature
        self.llm = LLMRails(rails_config, llm=llm)
        self.logger = logger
        
    
    async def generate_answer(self, query, docs):
        start_time =time.time()
        messages = [{"role": "system", "content": system_prompt_generator},
                    {"role": "user", "content": get_user_prompt_generator(query,docs)}]
        
        #Count input tokens for logs, metrics
        enc = tiktoken.encoding_for_model(self.model_name)  # or your model
        def count_tokens(text):
            return len(enc.encode(text))
        input_tokens = sum(count_tokens(m["content"]) for m in messages)
        
        #Stream output
        output = ""
        first_token_time = None
        async for chunk in self.llm.stream_async(messages=messages):
            if first_token_time is None:
                first_token_time = time.time()
            output += chunk
            yield chunk

        #Count output tokens + logs
        output_tokens = count_tokens(output)
        self.logger.info("llm_completed", 
            model_name=self.model_name, 
            temperature=self.temperature, 
            llm_latency_ms=1000*(time.time()-start_time), 
            llm_latency_firsttoken_ms = 1000*(first_token_time-start_time),
            input_tokens = input_tokens,
            output_tokens=output_tokens)
