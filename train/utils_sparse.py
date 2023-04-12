# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:35:41 2021
"""

import numpy as np
import faiss

class FaissKNeighbors:
    '''
    "Make kNN 300 times faster than Scikit-learn’s in 20 lines! -
    Using Facebook faiss library for REALLY fast kNN" - by Jakub Adamczyk

    url: https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
    '''
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

def normc(data):
    return data / np.linalg.norm(data, axis=1, keepdims=True)