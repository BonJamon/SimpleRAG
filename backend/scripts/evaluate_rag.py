import pandas as pd
import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.rag.evaluation import full_evaluation
from app.rag.retriever import RetrieverV1
from app.rag.generator import GeneratorV1
from dotenv import load_dotenv

load_dotenv()

test_data = pd.read_csv("data/synthetic_test_data.csv",encoding="cp1252").iloc[:2]

retriever = RetrieverV1(k=3)
generator = GeneratorV1(temperature=0.1, model_name="gpt-4o-mini")

performances = asyncio.run(full_evaluation(test_data, retriever, generator))

print(performances)