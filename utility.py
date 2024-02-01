#type hints
import pickle
from typing import Dict, List, Tuple

#data loading
from beir.datasets.data_loader import GenericDataLoader
from beir import util
import pathlib, os

#progress bar
from tqdm.notebook import tqdm

#beir dense retrieval and beir evaluation 
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#text cleaning
import spacy

#multiprocessing
from multiprocessing import Pool, cpu_count

#sparse retrieval
from rank_bm25 import BM25Okapi

#fast top k/top k prime
import heapq

import numpy as np
import matplotlib.pyplot as plt

def data_preparation(dataset: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Download the given dataset from beir.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[Dict[str, Dict[str, str]], Dict[str, str]]: documents and queries.
    """
    
    #Download dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(os.path.abspath('')), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    
    #Retrieve documents and queries
    documents,queries,_=GenericDataLoader(data_path).load(split="test")
    
    return documents,queries

#load the spacy model for lemmatization 
_nlp = spacy.load("en_core_web_lg",disable=['parser','ner'])

#text cleaning and tokenization function
_tokenizer_cleaner = lambda text: [token.lemma_ for token in _nlp(text) if not token.is_stop and not token.is_punct]

def _clean_document(document: Tuple[str, Dict[str, str]]) -> Tuple[str, str, str]:
    """
    Auxiliary function for cleaning and tokenize the given document.
    
    Args:
        document (Tuple[str, Dict[str, str]]): tuple of document id and dict of document text and document title

    Returns:
        Tuple[str, str, str]: tuple of document id, cleaned and tokenized document text and cleaned and tokenized document title
    """
    id, doc_old = document
    
    return id, _tokenizer_cleaner( doc_old["title"] ), _tokenizer_cleaner( doc_old["text"] )

def _BM25(bm25, d_keys: List[str], q :List[Tuple[str, List[str]]], start:int, stop:int, skip:int) -> Dict[str, Dict[str, float]]:
    """
    Auxiliary function for the BM25 scores calculation.

    Args:
        bm25 (bm25 object): bm25 object
        d_keys (List[str]): list of document ids
        q (List[Tuple[str, List[str]]]): list of tuple of query id and tokenized query text
        start (int): start query index to process
        stop (int): stop query index to process
        skip (int): how many queries to skip

    Returns:
        Dict[str, Dict[str, float]]: dict of query id and dict of document id and relative score
    """
    
    results={}
    
    for i in range(start, stop, skip):
        
        q_id, query = q[i]
        
        scores=bm25.get_scores(query)
        
        #documents which score is 0 are not present for performance reasons
        results[q_id]={ key:score for key,score in zip(d_keys,scores) if score!=0 }
        
    return results
    
def sparse_retrieval(documents: Dict[str, Dict[str, str]], queries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    BM25 scores calculation.

    Args:
        documents (Dict[str, Dict[str, str]]) 
        queries (Dict[str, str])

    Returns:
        Dict[str, Dict[str, float]]: dict of query id and dict of document id and relative score.
    """
    
    #document & query cleaning and tokenization
    
    #parallel document cleaning and tokenization
    with Pool() as p:
        tokenized_docs=list(tqdm( p.imap(_clean_document, documents.items()), 
                                            total=len(documents),
                                            desc="documents cleaning and tokenization"))
    d={}
    for id, text, title in tokenized_docs:
        d[id]=title+text
    
    #query cleaning
    q={}
    for id,text in tqdm( queries.items(), desc="queries cleaning and tokenization" ):
        q[id]=_tokenizer_cleaner( text )
    

    #BM25
    
    results={}
    
    #progress bar
    pbar = tqdm(total=len(q),desc="BM25 scores calculation")
    
    #callback execute at the end of each execution
    def callback(result):
        
        #update results with intermediate result
        results.update(result)
        
        #update progress bar
        pbar.update(len(result))
    
    #process pool
    p=Pool()
    
    #process variables
    bm25 = BM25Okapi(d.values())
    d_keys = list(d.keys())
    q_items = list(q.items())
    
    #processes start
    for start in range( cpu_count() ):
        p.apply_async(func=_BM25,args=(bm25, d_keys, q_items, start, len(q), cpu_count(), ), callback=callback, error_callback = lambda x: print(x))
        
    p.close()
    p.join()
    pbar.close()
    
    return results

def dense_retrieval(documents: Dict[str, Dict[str, str]], queries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Dense scores calculation using "all-MiniLM-L6-v2" model.

    Args:
        documents (Dict[str, Dict[str, str]])
        queries (Dict[str, str])

    Returns:
        Dict[str, Dict[str, float]]: dict of query id and dict of document id and relative score.
    """
    
    model = DRES(models.SentenceBERT("all-MiniLM-L6-v2"))
    retriever = EvaluateRetrieval(model, k_values=[len(documents)], score_function="dot")

    results = retriever.retrieve(documents, queries)
    
    return results

def ground_truth(results_sparse: Dict[str, Dict[str, float]], results_dense: Dict[str, Dict[str, float]],
                k:int ) -> Dict[str, Dict[str, int]]:
    """
    Calculation of the ground truth for each query by:
    1) summing up the BM25 score and the dense score for each document
    2) retrieve the top k documents

    Args:
        results_sparse (Dict[str, Dict[str, float]]): dict of query id and dict of document id and relative sparse score.
        results_dense (Dict[str, Dict[str, float]]): dict of query id and dict of document id and relative dense score.
        k (int): number of document to take into account for the top-k.

    Returns:
        Dict[str, Dict[str, int]]: dict of query id and dict of document id and relative score.
    """
    
    real_result={}

    #for each query iterate over sparse and dense documents
    for quey_id, relevant_sparse in results_sparse.items():
        
        relevant_dense=results_dense[quey_id]
        
        #union of sparse documents and dense documents by summing up the scores
        #if id not present the corresponding score is assumed:
        # -to be 0 for the sparse representation.
        # -to be 0 for the dense representation.
        documents_per_query={ doc_id: relevant_sparse.get(doc_id, 0) + relevant_dense.get(doc_id, 0)  
                                for doc_id in set(relevant_sparse) | set(relevant_dense) }
        
        #top k relevant document ids
        top_k_documents = heapq.nlargest(k, documents_per_query, key=documents_per_query.get)
        
        #the score for each relevant document is set to 1
        real_result[quey_id]={ key:1 for key in top_k_documents }
    
    return real_result

def merging(results_sparse: Dict[str, Dict[str, float]],
            results_dense: Dict[str, Dict[str, float]], k_prime:int, k:int) -> Dict[str, Dict[str, float]]:
    """
    Calculation of the merging results for each query by:
    1) retrieve the top k' documents from the sparse and dense results.
    2) summing up the top k' sparse documents scores and the top k' dense documents scores

    Args:
        results_sparse (Dict[str, Dict[str, float]]): dict of query id and dict of document id and relative sparse score.
        results_dense (Dict[str, Dict[str, float]]): dict of query id and dict of document id and relative dense score.
        k_prime (int): number of document to take into account for the top-k'.

    Returns:
        Dict[str, Dict[str, int]]: dict of query id and dict of document id and relative score.
    """
    
    result={}
    
    #for each query iterate over sparse and dense documents
    for quey_id, relevant_sparse in results_sparse.items():
        
        relevant_dense=results_dense[quey_id]
        
        #top k prime sparse document ids
        top_k_prime_documents_sparse = heapq.nlargest(k_prime, relevant_sparse, key=relevant_sparse.get)

        #top k prime dense document ids
        top_k_prime_documents_dense = heapq.nlargest(k_prime, relevant_dense, key=relevant_dense.get)
        
        #union of top k prime sparse documents and top k prime dense documents by summing up their scores
        #if id not present the corresponding score is assumed:
        # -to be 0 for the sparse representation.
        # -to be 0 for the dense representation.
        merged = { doc_id: relevant_sparse.get(doc_id, 0) + relevant_dense.get(doc_id, 0)
                            for doc_id in set(top_k_prime_documents_sparse) | set(top_k_prime_documents_dense) }
        
        #retrieve the top-k documents from the merged set
        top_k_documents_merged = heapq.nlargest(k, merged, key=merged.get)
        
        #id : merged_score
        result[quey_id] = { doc_id: merged[doc_id] for doc_id in set(top_k_documents_merged) }
        
    return result
    
def metrics_calculation(dataset:str,results_sparse: Dict[str, Dict[str, float]], results_dense: Dict[str, Dict[str, float]],
                        ks: list[int] = [50,100,150], k_primes: list[int] = [x for x in range(20, 170, 5)]) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Metrics calculation (ndcg, recall and precision) for each values of k' by fixing the k value. 

    Args:
        results_sparse (Dict[str, Dict[str, float]]): dict of query id and dict of document id and relative sparse score.
        results_dense (Dict[str, Dict[str, float]]): dict of query id and dict of document id and relative dense score.
        ks (list[int], optional): list of k values for calculating the ground truth. Defaults to [50,100,150].
        k_primes (list[int], optional): list of k' values for calculating the merged results for each fixed k. Defaults to [x for x in range(20, 170, 5)].

    Returns:
        Dict[int, Dict[int, Dict[str, float]]]: dict of k value and dict of k' value and dict of metric name and metric value.
    """

    metrics_per_k={}
    
    #for each fixed value of k:
    for k in tqdm(ks, desc="k values:"):
        metrics_per_k_prime = {}

        #fixed k ground truth calculation
        ground_truth_k = ground_truth(results_sparse, results_dense, k)
        
        #for each value of k':
        for k_prime in tqdm( k_primes, desc="k' values:"):
            
            #merging with the given k'
            results = merging(results_sparse, results_dense, k_prime, k)
            
            #metric calculation with given k ground truth and k' merging
            ndcg, _, recall, precision = EvaluateRetrieval.evaluate(ground_truth_k,
                                                                    results, [k])

            #better formatting for the plot function
            metrics = {"ndcg": list(ndcg.values())[0], "recall": list(recall.values())[0], "precision": list(precision.values())[0]}

            metrics_per_k_prime[k_prime] = metrics
        
        metrics_per_k[k]=metrics_per_k_prime
        
    with open("metrics/"+dataset+"_metrics.pkl", 'wb') as outp:
        pickle.dump(metrics_per_k, outp, pickle.HIGHEST_PROTOCOL)
        
    return metrics_per_k

def load_metrics(dataset:str) -> Dict[int, Dict[int, Dict[str, float]]] :
    """
    Load the metrics for the given dataset.

    Args:
        dataset (str): file name to load.

    Returns:
        Dict[int, Dict[int, Dict[str, float]]]: metrics.
    """
    
    with open(dataset+"_metrics.pkl", 'rb') as inp:
        metrics_per_k = pickle.load(inp)
    
    return metrics_per_k

def plot_top_k_metrics_vs_k_prime(dataset_name:str, metrics_per_k: Dict[int, Dict[int, Dict[str, float]]]):
    """
    Results plot function.

    Args:
        dataset_name (str): dataset name for plot title.
        metrics_per_k (Dict[int, Dict[int, Dict[str, float]]]): metrics to plot.
    """

    for metric_name in ["ndcg", "recall", "precision"]:

        fig, ax = plt.subplots(figsize=(15,10))
        
        ax.set_xlabel("k'",fontsize=15)
        ax.set_ylabel(metric_name,fontsize=15)
        
        ax.set_title(dataset_name+": "+metric_name+" vs k'",fontsize=20)

        for k, metrics_per_k_prime in metrics_per_k.items():
            
            x, y = np.array([[key, metrics[metric_name]]
                            for key, metrics in metrics_per_k_prime.items()]).T

            ax.plot(x, y, '-x', label="k = "+str(k))
        
        ax.xaxis.set_ticks(np.arange(15, 175, 5.0))
        ax.yaxis.set_ticks(np.arange(0, 1.05, 0.05))
        
        ax.grid()
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.93, 0.93))