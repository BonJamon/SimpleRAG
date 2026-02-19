from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from templates import system_prompt, get_user_prompt
import asyncio
from helper import catch_streaming_response, get_retriever

load_dotenv()



async def retrieve_docs(query,retriever,k=3):
    contexts = await retriever.ainvoke(query)
    return contexts[:k]


async def generate_answer(query, docs, temperature=0.1, model_name = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=300, streaming=True)

    messages = [("system", system_prompt),
                ("human", get_user_prompt(query,docs))]
    
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content
    
    #return await llm.ainvoke(messages).content


async def answer(query, docs, k=2, temperature=0.1, model_name="gpt-4o-mini"):
    docs = await retrieve_docs(query, k)
    docs_clean = [doc.page_content for doc in docs]

    return await generate_answer(query, docs_clean, temperature, model_name)



if __name__=="__main__":
    query = "How to fix my router?"
    retriever = asyncio.run(get_retriever())
    docs = asyncio.run(retrieve_docs(query, retriever, k=2))
    docs_clean = [doc.page_content for doc in docs]
    print(asyncio.run(catch_streaming_response(generate_answer, query, docs_clean)))