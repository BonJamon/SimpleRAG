system_prompt_generator = """You are a factual question-answering assistant.

Answer the question using only the provided context.

Guidelines:
- Base your answer strictly on the provided context.
- If the answer is explicitly stated or clearly described in the context, provide it.
- You may restate or summarize information from the context.
- Do not introduce information not present in the context.
- Cite supporting document IDs in parentheses.
- Only respond with:
  "The provided context does not contain enough information to answer this question."
  if the answer is completely absent from the context.
- If the context contains conflicting information, state that there is conflicting information in the provided context.
- Be concise and precise."""

def convert_doc_text(doc, id):
    """
    Convert document str into pure text for the prompt
    
    :param docs: List[Tuple] containing Id and text
    """
    return f"""Document ID: {id}
    Content: 
    {doc}"""

def construct_context(docs):
    context = ""
    for i in range(len(docs)):
        doc = docs[i]
        context += convert_doc_text(doc,i)+"\n\n"

    return context
    

def get_user_prompt_generator(user_query, retrieved_documents):
    
    context = construct_context(retrieved_documents)
    user_prompt = f"""Question:
{user_query}

Retrieved Context:
{context}
Answer:"""
    return user_prompt




system_prompt_standalone = """You are a standalone question generator that takes a question and previous conversation history 
to generate a rephrased question that contains all necessary information to stand alone.

Rules:
- Make this question as short as possible without omitting relevant information
"""

def get_user_prompt_standalone(question, conversation):
    return f"""Question: {question}


Conversation history:
{conversation}
"""