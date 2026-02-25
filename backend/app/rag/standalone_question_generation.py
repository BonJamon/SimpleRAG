from langchain_openai import ChatOpenAI
from app.rag.templates import system_prompt_standalone, get_user_prompt_standalone
from abc import ABC, abstractmethod


class StandaloneQuestionGenerator(ABC):
    @abstractmethod
    async def generate_standalone_question(self, question, conversation):
        pass

class StandaloneQuestionGeneratorV1:
    def __init__(self, model_name, temperature, logger):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=200)
        self.logger = logger

    async def generate_standalone_question(self, question, conversation):
        messages = [("system", system_prompt_standalone),
                     ("human", get_user_prompt_standalone(question, conversation))]
        response = await self.llm.ainvoke(messages)
        response = response.content
        self.logger.info("standalone_question_generated", 
                         standalone_question_length=len(response), 
                         standalone_generator_model_name = self.llm.model_name, 
                         standalone_generator_temperature = self.llm.temperature)
        return response
