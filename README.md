# RAG-PopQA-Assignment
Retrieval-Augmented Generation project using PopQAv

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for factual question answering using the PopQA dataset. The system retrieves relevant passages and generates grounded answers with citations, followed by a self-reflective step to improve answer quality.

---

## Features

* Dense retrieval using sentence embeddings and FAISS
* Query expansion to improve retrieval coverage
* Hybrid search combining BM25 and dense retrieval
* Reranking using a cross-encoder model
* Citation-grounded answer generation
* Self-reflective component for answer refinement

---

## Dataset

* Benchmark: PopQA dataset
* Corpus:

  * General text corpus (AG News dataset)
  * Answer-based passages extracted from PopQA

A hybrid corpus is used to balance realism and retrieval accuracy.

---

## Project Structure

```id="struct1"
rag_project/
│
├── main.py
├── dataset.py
├── retrieval.py
├── hybrid.py
├── reranker.py
├── generator.py
├── evaluation.py
├── query_expansion.py
├── reflection.py
```

---

## How to Run

### 1. Install dependencies

```id="install1"
pip install datasets sentence-transformers transformers faiss-cpu rank-bm25
```

### 2. Run the project

```id="run1"
python main.py
```

---

## Evaluation Metrics

The system is evaluated using:

* Recall@k
* Precision@k
* Mean Reciprocal Rank (MRR)

These metrics measure retrieval accuracy and ranking effectiveness.

---

## Output

The system produces:

* Retrieval comparison across multiple configurations
* Evaluation metrics for each method
* Generated answers with citations
* Refined answers after self-reflection

---

## Limitations

* The corpus is not a full Wikipedia dataset
* Retrieval performance depends on corpus quality
* Some answers may be incomplete or unsupported

---

## Future Work

* Use a full Wikipedia corpus
* Improve query expansion using language models
* Enhance reranking and retrieval fusion strategies
* Improve grounding and citation accuracy

---


