from langchain_community.vectorstores import FAISS
from embeddings import E5Embeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from templates import system_prompt, get_user_prompt

load_dotenv()

def retrieve_docs(query,k=3):
    embeddings = E5Embeddings()
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    return retriever.invoke(query)[:k]


def generate_answer(query, docs, temperature=0.1, model_name = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=300)

    messages = [("system", system_prompt),
                ("human", get_user_prompt(query,docs))]
    

    return llm.invoke(messages).content


def answer(query, docs, k=2, temperature=0.1, model_name="gpt-4o-mini"):
    docs = retrieve_docs(query, k)
    docs_clean = [(doc.id, doc.page_content) for doc in docs]
    return generate_answer(query, docs_clean, temperature, model_name)



if __name__=="__main__":
    query = "How to fix my router?"
    docs = retrieve_docs(query)[:2]
    docs_clean = [doc.page_content for doc in docs]
    print(generate_answer(query, docs_clean))