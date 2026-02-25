from app.services.utils import load_conversation, save_conversation
from app.rag.pipeline import Pipeline
from app.rag.standalone_question_generation import StandaloneQuestionGenerator
from app.models import Question

class RAG:
    def __init__(self, pipeline:Pipeline, standalone_question_generator:StandaloneQuestionGenerator):
        self.pipeline = pipeline
        self.standalone_question_generator = standalone_question_generator

    
    async def stream_answer(self, question:Question, session_id:str):
        #Construct standalone question (Relevant for follow up questions)
        standalone_question = await self._construct_standalone_question(question)
        #Generate answer
        answer = ""
        async for chunk in self.pipeline.stream_answer(standalone_question=standalone_question, original_question=question.question):
            yield f"data: {chunk}\n\n"
            answer += chunk
        #Save answer
        save_conversation(question.question, answer, session_id)


    async def _construct_standalone_question(self, question:Question):
        #If no previous convo: Nothing to do
        if question.session_id is None:
            return question.question
        #If previous convo: Summarize into new question for rag
        conversation = load_conversation(question.session_id)
        conversation = conversation[-min(len(conversation), 5):]
        return await self.standalone_question_generator.generate_standalone_question(question.question, conversation)



            