from pdb import set_trace
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from time import time

def term_frequency_vectorizer():
    vectorizer = TfidfVectorizer()
    return vectorizer

def ssim(query, target):
    return ssim(query, target)

def try_join(l):
    try:
        return ','.join(map(str, l))
    except TypeError:
        return np.nan


class IncomeTaxWordReprTFClassifier():
    def __init__(self, documents, labels) -> None:
        self.documents = documents
        self.labels = labels
        self.spans = self.labels.unique().tolist()
        self.vectorizers = [ TfidfVectorizer() for _ in self.spans ]
        self.document_class = []
        for label in self.spans:
            print(f"Converting label {label}", end=' ')
            st = time()
            self.document_class.append(
                self.documents[self.labels[self.labels == label].index].tolist())
            self.vectorizers[label].fit(self.document_class[-1])
            et = time()
            print(f"done takes {(et - st)/1000:.3f}ms")
        assert len(self.document_class) == len(self.labels.unique())

    def compute_mean_tfidf_score(self, document, label):
        current_doc_vec = self.vectorizers[label].transform(
            [document] + self.document_class[label])
        
        cos_score = linear_kernel(
            current_doc_vec[0:1], 
            current_doc_vec).flatten()
        document_score = [ (item.item()) 
            for (item) in cos_score[1:]]
        document_score = list(reversed(document_score))

        return sum(document_score )/ len(self.document_class[label])
    
    def __call__(self, document) :
        # compute tf idf average score for comparing the current document against class of the document.
        scores_among_labels = []
        for label in self.spans:
            scores_among_labels.append( 
                self.compute_mean_tfidf_score(document, label)
            )
        return scores_among_labels
    
