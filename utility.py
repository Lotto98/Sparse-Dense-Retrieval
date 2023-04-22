from beir.datasets.data_loader import GenericDataLoader
from beir import util

from tqdm.notebook import tqdm

import pathlib, os

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import spacy
from multiprocessing import Pool

import warnings

import heapq

import numpy as np

from rank_bm25 import BM25Okapi

_nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "parser", "attribute_ruler", "ner"])

_tokenizer_cleaner = lambda text: [token.lemma_ for token in _nlp(text) if not token.is_stop and not token.is_punct]

def data_preparation(dataset:str):
       
       # Download dataset and unzip the dataset
       url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
       out_dir = os.path.join(pathlib.Path(os.path.abspath('')), "datasets")
       data_path = util.download_and_unzip(url, out_dir)
       
       documents,queries,_=GenericDataLoader(data_path).load(split="test")
       
       return documents,queries

def _clean_document(document):
        
    id, doc_old = document

    return id, _tokenizer_cleaner( doc_old["title"] ), _tokenizer_cleaner( doc_old["text"] )

def _BM25(bm25, d_keys, q, start, stop, skip):
    
    results={}
    
    for i in range(start, stop, skip):
        
        q_id, query = q[i]
        
        scores=bm25.get_scores(query)
        
        #documents which score is 0 are not present for performance reasons
        results[q_id]={ key:score for key,score in zip(d_keys,scores) if score!=0 }
        
    return results
    
def BM25_retrieval(documents,queries):
    
    #document & query cleaning and tokenization
    
    #disable warning for the nlp
    with warnings.catch_warnings():
            
        warnings.simplefilter("ignore")
        
        #parallel document cleaning and tokenization
        with Pool(8) as p:
            tokenized_docs=list(tqdm( p.imap(_clean_document, documents.items()), 
                                                total=len(documents),
                                                desc="document cleaning and tokenization"))
        d={}
        for id, text, title in tokenized_docs:
            d[id]=title+text
        
        #query cleaning
        q={}
        for id,text in tqdm( queries.items(), desc="query cleaning and tokenization" ):
            q[id]=_tokenizer_cleaner( text )
    
    #BM25
    
    results={}
    
    #progress bar
    pbar = tqdm(total=len(q),desc="BM25")
    
    #callback execute at the end of each execution
    def callback(result):
        
        #update results with intermediate result
        results.update(result)
        
        #update progress bar
        pbar.update(len(result))
    
    #process pool
    p=Pool(8)
    
    #process variables
    bm25=BM25Okapi(d.values())
    d_keys = list(d.keys())
    q_items = list(q.items())
    
    #processes start
    for start in range(8):
        p.apply_async(func=_BM25,args=(bm25, d_keys, q_items, start, len(q), 8, ), callback=callback, error_callback = lambda x: print(x))
        
    p.close()
    p.join()
    
    return results

def dense_retrieval(documents, queries):
    
    model = DRES(models.SentenceBERT("all-MiniLM-L6-v2"))
    retriever = EvaluateRetrieval(model, k_values=[len(documents)], score_function="dot")

    results = retriever.retrieve(documents, queries)
    
    return results

def ground_truth(results_sparse, results_dense, k:int ):
    
    real_result={}

    #for each query iterate over sparse and dense documents
    for ( quey_id, relevant_sparse ), relevant_dense in zip( results_sparse.items(), results_dense.values() ):
        
        #union of sparse documents and dense documents by summing up the scores
        #if id not present the corresponding score is assumed:
        # -to be 0 for the sparse representation.
        # -to be -inf for the dense representation.
        documents_per_query={ doc_id: relevant_sparse.get(doc_id, 0) + relevant_dense.get(doc_id, float("-inf"))  
                                for doc_id in set(relevant_sparse) | set(relevant_dense) }
        
        #top k relevant document ids
        k_keys_sorted_by_values = heapq.nlargest(k, documents_per_query, key=documents_per_query.get)
        
        #the score for each relevant document is set to 1
        real_result[quey_id]={ key:1 for key in k_keys_sorted_by_values }
    
    return real_result

def merging( results_sparse, results_dense, k_prime:int ):
    
    result={}
    
    #for each query iterate over sparse and dense documents
    for ( quey_id, relevant_sparse ), relevant_dense in zip( results_sparse.items(), results_dense.values() ):
        
        #top k prime sparse document ids
        top_k_prime_documents_sparse = heapq.nlargest(k_prime, relevant_sparse, key=relevant_sparse.get)

        #top k prime dense document ids
        top_k_prime_documents_dense = heapq.nlargest(k_prime, relevant_dense, key=relevant_dense.get)
        
        #union of top k prime sparse documents and top k prime dense documents by summing up their scores
        #if id not present the corresponding score is assumed:
        # -to be 0 for the sparse representation.
        # -to be -inf for the dense representation.
        result[quey_id]={ doc_id: relevant_sparse.get(doc_id, 0) + relevant_dense.get(doc_id, float("-inf"))
                            for doc_id in set(top_k_prime_documents_sparse) | set(top_k_prime_documents_dense) }
        
    return result