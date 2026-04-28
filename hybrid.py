from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, corpus, dense_model, index):
        self.corpus = corpus
        self.tokenized = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized)
        self.dense_model = dense_model
        self.index = index

    def search(self, query, k=5):
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]

        q_emb = self.dense_model.encode([query])
        _, dense_ids = self.index.search(q_emb, k)

        combined = list(set(bm25_top + list(dense_ids[0])))
        return [(f"P{i}", self.corpus[i][:200]) for i in combined]