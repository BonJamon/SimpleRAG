from typing import Optional
from nemoguardrails.actions import action
from nemoguardrails.llm.taskmanager import LLMTaskManager
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

@action(is_system_action=True, execute_async=True)
async def classify_intent(context: Optional[dict] = None):
    """Check if bot response contains blocked terms."""
    user_message = context.get("user_message", "")

    system_prompt = f"""Your task is to classify the user message into jailbreak_attempt, support_request, other.
      Answer only with jailbreak_attempt/support_request/other"""
    
    messages = [("system", system_prompt),
                ("human", user_message)]
     
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=100)
    result = await llm.ainvoke(messages)
    return result.content

@action(is_system_action=True, execute_async=True)
async def check_scope(context: Optional[dict] = None, **kwargs):
    """Check if bot response contains blocked terms."""
    bot_message = context.get("bot_message", "")
    #llm_task_manager = kwargs.get("llm_task_manager")

    system_prompt = f"""Your task is to check whether the bot output contains any harmful language.
    Answer with yes/no"""
    
    messages = [("system", system_prompt),
                ("assistant", bot_message)]
     
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=100)
    result = await llm.ainvoke(messages)
    return result.content


    


    blocked_terms = ["confidential", "proprietary", "secret"]

    for term in blocked_terms:
        if term in bot_response.lower():
            return True  # Term found, block the response

    return False  # No blocked terms found