import pandas as pd
from query import retrieve_docs, generate_answer
import re
from typing import List
import ast
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
import asyncio
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

"""
Helper Functions
"""

def normalize_ground_truth(gt_chunks: List[str]) -> List[str]:
    """
    Removes <n-hop> markers and leading whitespace.
    """
    cleaned = []
    for chunk in gt_chunks:
        # Remove <n-hop>\n\n prefix
        chunk = re.sub(r"<\d+-hop>\s*", "", chunk)
        cleaned.append(chunk.strip())
    return cleaned

def normalize_retrieved(retrieved_chunks: List[str]) -> List[str]:
    """
    Basic normalization for retrieved chunks.
    """
    return [c.page_content.strip() for c in retrieved_chunks]


def exact_match_retrieval(ground_truth: List[str], retrieved: List[str]):
    gt_clean = normalize_ground_truth(ground_truth)
    ret_clean = normalize_retrieved(retrieved)

    matches = []
    first_hit=None
    for i in range(len(ret_clean)):
        r = ret_clean[i]
        for gt in gt_clean:
            if r == gt:
                matches.append(r)
                if first_hit is None:
                    first_hit=i+1

    return matches, first_hit


"""
Performances per row
"""

def get_retrieval_performance(true_context, retrieved_context,k):
    """
    Docstring for get_retrieval_performance
    
    :param true_context: List[str]
    :param retrieved_context: List[str]
    :param k: number of retrieved results
    :returns: Dict containing performances
    """
    matches, first_hit = exact_match_retrieval(true_context, retrieved_context)
    performances = {}
    performances["precision"] = len(matches)/k
    performances["recall"] = len(matches)/len(true_context)
    performances["first_hit"] = first_hit
    performances["hit"] = (first_hit is not None)
    return performances


async def get_faithfullness(user_input, contexts, response):
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    # Create metric
    scorer = Faithfulness(llm=llm)

    # Evaluate
    result = await scorer.ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=contexts
    )
    return result



"""
Evaluations of components
"""

def evaluate_retrieval(test_data,k):
    precisions= []
    recalls= []
    first_hits = []
    for i in range(len(test_data)):
        #Retrieve Context
        user_input = test_data["user_input"].loc[i]
        retrieved_context = retrieve_docs(user_input,k)
        #Evaluate each retrieval
        true_context = ast.literal_eval(test_data["reference_contexts"].loc[i])
        performances = get_retrieval_performance(true_context, retrieved_context, k)
        precisions.append(performances["precision"])
        recalls.append(performances["recall"])
        first_hits.append(performances["first_hit"])
    #Average performances
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    #clean first hits from Nones
    cleaned_hits = []
    n_Nones = 0
    for hit in first_hits:
        if hit is not None:
            cleaned_hits.append(hit)
        else:
            n_Nones+=1
    avg_first_hit = sum(cleaned_hits) / len(cleaned_hits)
    return precision, recall, avg_first_hit, n_Nones



async def evaluate_generation(test_data,k, temperature=0.1, model_name="gpt-4o-mini"):
    """
    Compute faithfulness, Rouge, BlEU
    
    :param test_data: test dataset
    :param k: retrieved results
    """
    faithfullnesses = []
    for i in range(len(test_data)):
        user_input = test_data["user_input"].loc[i]
        docs = retrieve_docs(user_input, k)
        docs_tuple= [(doc.id, doc.page_content) for doc in docs]
        docs_text = [doc.page_content for doc in docs]
        response = generate_answer(user_input, docs_tuple, temperature, model_name)
        f = await get_faithfullness(user_input, docs_text, response)
        faithfullnesses.append(f)
    
    performances = {}
    performances["faithfullness"] = sum(faithfullnesses) / len(faithfullnesses)
    return performances


async def full_evaluation(test_data, k=3, temperature=0.1, model_name="gpt-4o-mini"):
    nltk.download('punkt_tab')
    performances = {}
    for i in range(len(test_data)):
        t1 = time.time()
        #Retrieval
        user_input = test_data["user_input"].loc[i]
        docs = retrieve_docs(user_input, k)
        #Generation
        docs_text = [doc.page_content for doc in docs]
        response = generate_answer(user_input, docs_text, temperature, model_name)
        response_speed = time.time()-t1

        #Get performances for query
        true_context = ast.literal_eval(test_data["reference_contexts"].loc[i])
        performances_retrieval = get_retrieval_performance(true_context, docs, k)
        faithfulness = await get_faithfullness(user_input, docs_text, response)

        reference_tokenized = [nltk.word_tokenize(test_data["reference"].loc[i])]
        candidate_tokenized = nltk.word_tokenize(response)

        bleu = sentence_bleu(reference_tokenized, candidate_tokenized)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(test_data["reference"].loc[i], response)
        rouge_1 = scores["rouge1"].fmeasure
        rouge_L = scores["rougeL"].fmeasure
        performances.setdefault("bleu",[]).append(bleu)
        performances.setdefault("rouge1", []).append(rouge_1)
        performances.setdefault("rougeL", []).append(rouge_L)
        performances.setdefault("faithfullness",[]).append(faithfulness)
        performances.setdefault("response_speed", []).append(response_speed)
        for key,val in performances_retrieval.items():
            if key != "first_hit" or (key == "first_hit" and val is not None): 
                #filter None's for first_hit
                performances.setdefault(key, []).append(val)
    
    #Average performances over queries
    for key,val in performances.items():
        performances[key] = sum(performances[key]) / len(performances[key]) if len(performances[key]) > 0 else 0
    return performances





if __name__=="__main__":
    test_data = pd.read_csv("data/synthetic_data.csv",encoding="cp1252").loc[:2]
    performances = asyncio.run(full_evaluation(test_data))
    print(performances)





