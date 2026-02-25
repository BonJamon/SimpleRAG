from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()


# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-mini")
#embeddings = E5Embeddings()
embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small",
        model_kwargs={"device": "cpu"},  # or "cpu"
        encode_kwargs={"normalize_embeddings": True}
    )

generator = TestsetGenerator.from_langchain(llm=generator_llm, embedding_model=embeddings)

#Get documents
chunk_dir = './data/processed/chunks/project-assurance-data'
docs = []
for chunk_file in os.listdir(chunk_dir):
    with open(os.path.join(chunk_dir, chunk_file), "r", encoding="utf-8") as f:
        chunk = f.read()
    docs.append(Document(chunk))



#Generate synthetic data
testset = generator.generate_with_langchain_docs(docs[:15], 30)

print(testset)
testset.to_csv("./data/synthetic_test_data.csv")