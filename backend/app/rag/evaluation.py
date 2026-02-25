import pandas as pd
import ast
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
import asyncio
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from app.rag.helper import catch_streaming_response, exact_match_retrieval
from app.rag.retriever import Retriever
from app.rag.generator import Generator


"""
Performances per row
"""

def get_retrieval_performance(true_context, retrieved_context):
    """
    Docstring for get_retrieval_performance
    
    :param true_context: List[str]
    :param retrieved_context: List[str]
    :param k: number of retrieved results
    :returns: Dict containing performances
    """
    matches, first_hit = exact_match_retrieval(true_context, retrieved_context)
    k = len(retrieved_context)
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

    try:
        # Evaluate
        result = await scorer.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=contexts
        )
    except Exception as exp:
        print(exp)
        result = None
        pass
    return result

     



"""
Evaluations of components
"""

async def evaluate_retrieval(test_data,retriever):
    precisions= []
    recalls= []
    first_hits = []
    for i in range(len(test_data)):
        #Retrieve Context
        user_input = test_data["user_input"].loc[i]
        retrieved_context = await retriever.retrieve_documents(user_input)
        #Evaluate each retrieval
        true_context = ast.literal_eval(test_data["reference_contexts"].loc[i])
        performances = get_retrieval_performance(true_context, retrieved_context)
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



async def evaluate_generation(test_data,retriever:Retriever, generator:Generator):
    """
    Compute faithfulness, Rouge, BlEU
    
    :param test_data: test dataset
    :param k: retrieved results
    """
    faithfullnesses = []
    times_to_first_token = []
    total_times = []
    for i in range(len(test_data)):
        user_input = test_data["user_input"].loc[i]
        docs = await retriever.retrieve_documents(user_input)
        docs_text = [doc.page_content for doc in docs]
        response, t1, t2, t3 = await catch_streaming_response(generator.generate_answer, user_input, docs_text)
        f = await get_faithfullness(user_input, docs_text, response)
        faithfullnesses.append(f)
        times_to_first_token.append(t2-t1)
        total_times.append(t3-t1)
    
    #clean faithfullnesses
    faithfullnesses = [f for f in faithfullnesses if f is not None]
    performances = {}
    performances["faithfullness"] = sum(faithfullnesses) / len(faithfullnesses)
    performances["time_to_first_token"] = sum(times_to_first_token) / len(times_to_first_token)
    performances["total_time"] = sum(total_times) / len(total_times)
    return performances

async def full_evaluation(test_data, retriever:Retriever, generator:Generator):
    nltk.download('punkt_tab')
    performances = {}
    for i in range(len(test_data)):
        t1 = time.time()
        #Retrieval
        user_input = test_data["user_input"].loc[i]
        docs = await retriever.retrieve_documents(user_input)
        #Generation
        docs_text = [doc.page_content for doc in docs]
        response,t_startg,t_first_g,t_last_g = await catch_streaming_response(generator.generate_answer,user_input, docs_text)
        response_speed = time.time()-t1
        time_to_first_token = t_startg - t1

        #Get performances for query
        true_context = ast.literal_eval(test_data["reference_contexts"].loc[i])
        performances_retrieval = get_retrieval_performance(true_context, docs)
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
        if faithfulness is not None:
            performances.setdefault("faithfullness",[]).append(faithfulness)
        performances.setdefault("response_speed", []).append(response_speed)
        performances.setdefault("time_to_first_token", []).append(time_to_first_token)
        for key,val in performances_retrieval.items():
            if key != "first_hit" or (key == "first_hit" and val is not None): 
                #filter None's for first_hit
                performances.setdefault(key, []).append(val)
    
    #Average performances over queries
    for key,val in performances.items():
        performances[key] = sum(performances[key]) / len(performances[key]) if len(performances[key]) > 0 else 0
    return performances





if __name__=="__main__":
    test_data = pd.read_csv("data/synthetic_test_data.csv",encoding="cp1252")
    performances = asyncio.run(full_evaluation(test_data).iloc[:2])
    print(performances)





