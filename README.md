Basic QA utilizing RAG on technical support documentation. Based on FAISS, langchain and ragas for evaluation.

Current performances on llm-generated test data using ragas:
Precision@3         |0.4
Recall@3            |0.83
MRR                 |1.05
faithfullness       |0.88
BLEU                |0.28
ROUGE-1             |0.62
ROUGE-L             |0.53
response_time       |3.2s
time_to_first_token |0.06s


