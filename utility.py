import time
from beir.datasets.data_loader import GenericDataLoader
from beir import util

from tqdm.notebook import tqdm

import pathlib, os

from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import spacy
from multiprocessing import Pool
from subprocess import Popen, DEVNULL

import warnings

import heapq

import numpy as np

_nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "parser", "attribute_ruler", "ner"])

_cleaning = lambda text: " ".join( [token.lemma_ for token in _nlp(text) if not token.is_stop and not token.is_punct] )

def data_preparation(dataset:str):
       
       # Download dataset and unzip the dataset
       url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
       out_dir = os.path.join(pathlib.Path(os.path.abspath('')), "datasets")
       data_path = util.download_and_unzip(url, out_dir)
       
       documents,queries,_=GenericDataLoader(data_path).load(split="test")
       
       return documents,queries

def _clean_document(document):
        
    id, doc_old = document

    doc_new={
            "title": _cleaning( doc_old["title"] ),
            "text": _cleaning( doc_old["text"] )    
    }

    return id, doc_new
        
def BM25_retrieval(documents,queries):
        
        print("cleaning...")
        
        with warnings.catch_warnings():
                
                warnings.simplefilter("ignore")
                with Pool(8) as p:
                        d={id:doc for id,doc in list(tqdm( p.imap(_clean_document, documents.items()), 
                                                           total=len(documents),
                                                           desc="document cleaning"))}
                
                q={}
                for id,text in tqdm(queries.items(), desc="query cleaning"):
                        q[id]=_cleaning( text )
        
        print("BM25...")
        
        p=Popen(["elasticsearch-8.7.0/bin/elasticsearch"], stdout=DEVNULL)
        time.sleep(30)
        
        hostname = "http://elastic:sjI=G_r_Gyd+afe42LJ+@localhost:9200/"
        index_name = "bm25" 
        initialize = True
        number_of_shards=1

        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
        retriever = EvaluateRetrieval(model,score_function="dot")

        results = retriever.retrieve(d, q)
        
        p.terminate()
        
        return results

def dense_retrieval(documents,queries):
    model = DRES(models.SentenceBERT("all-MiniLM-L6-v2"))
    retriever = EvaluateRetrieval(model, score_function="dot")

    results = retriever.retrieve(documents, queries)
    
    return results

def ground_truth(results_sparse, results_dense, k:int ):
    
    real_result={}

    for (quey_id,relevant_sparse),relevant_dense in zip(results_sparse.items(),results_dense.values()):
        
        relevant_documents_per_query={key: relevant_sparse.get(key, 0) + relevant_dense.get(key, 0) 
                        for key in set(relevant_sparse) | set(relevant_dense)}
        
        k_keys_sorted_by_values = heapq.nlargest(k, relevant_documents_per_query, key=relevant_documents_per_query.get)
        
        real_result[quey_id]={ key:1 for key in k_keys_sorted_by_values }
    
    return real_result

def merging( results_sparse, results_dense,k_prime:int ):
    
    result={}
    
    for (quey_id,relevant_sparse),relevant_dense in zip( results_sparse.items(), results_dense.values() ):
        
        top_k_prime_documents_sparse_with_score = heapq.nlargest(k_prime, relevant_sparse.items(), key=lambda i: i[1])
    
        top_k_prime_documents_dense_with_score = heapq.nlargest(k_prime, relevant_dense.items(), key=lambda i: i[1])
        
        top_documents_merged=heapq.merge(top_k_prime_documents_sparse_with_score,top_k_prime_documents_dense_with_score,key=lambda i: i[1])
        
        result[quey_id]={}
        
        for (doc_id,score) in top_documents_merged:
            
            if doc_id not in result[quey_id]:
                result[quey_id][doc_id] = score
            else:
                #bho(?)
                #print("aia")
                result[quey_id][doc_id] += score
        
    return result