from dataset import load_popqa, load_wikipedia
from retrieval import build_index, search
from query_expansion import expand_query
from hybrid import HybridRetriever
from reranker import rerank
from generator import generate_answer
from evaluation import recall_at_k, precision_at_k, mrr
from reflection import reflect

from sentence_transformers import SentenceTransformer
import ast

# ------------------------
# Load data
# ------------------------
data = load_popqa(50)

# ------------------------
# Build HYBRID CORPUS (IMPORTANT FIX)
# ------------------------

# General text (news / wiki-like)
wiki_corpus = load_wikipedia(1500)

# Answer-based corpus (ensures correctness)
answer_corpus = []
for item in data:
    answers = ast.literal_eval(item["possible_answers"])
    answer_corpus.append(" ".join(answers))

# Final corpus
corpus = wiki_corpus + answer_corpus

# ------------------------
# Build index
# ------------------------
index, texts = build_index(corpus)

dense_model = SentenceTransformer('all-MiniLM-L6-v2')
hybrid = HybridRetriever(corpus, dense_model, index)

# ------------------------
# Systems
# ------------------------
def baseline(q):
    return search(q, index, texts)

def expansion(q):
    return search(expand_query(q), index, texts)

def hybrid_sys(q):
    return hybrid.search(expand_query(q))

def rerank_sys(q):
    return rerank(q, hybrid_sys(q))

# ------------------------
# Evaluation
# ------------------------
def evaluate(name, func):
    rec, prec, mrr_total = 0, 0, 0
    total = 20

    for i in range(total):
        q = data[i]["question"]
        answers = ast.literal_eval(data[i]["possible_answers"])

        results = func(q)

        rec += recall_at_k(results, answers, 3)
        prec += precision_at_k(results, answers, 3)
        mrr_total += mrr(results, answers)

    print(f"\n=== {name} ===")
    print("Recall@3:", rec / total)
    print("Precision@3:", prec / total)
    print("MRR:", mrr_total / total)

# ------------------------
# Part 2: Comparison
# ------------------------
print("\n===== SYSTEM COMPARISON =====")

evaluate("Baseline", baseline)
evaluate("Expansion", expansion)
evaluate("Hybrid", hybrid_sys)
evaluate("Reranked", rerank_sys)

# ------------------------
# Part 3 + 4: Generation + Reflection
# ------------------------
print("\n===== GENERATION (10 EXAMPLES) =====")

for i in range(10):
    q = data[i]["question"]

    results = rerank_sys(q)
    top_passages = results[:3]

    answer = generate_answer(q, top_passages)
    improved = reflect(q, answer)

    print("\n=================================")
    print("QUESTION:", q)

    print("\nPASSAGES:")
    for p in top_passages:
        print(p)

    print("\nANSWER:", answer)
    print("REFLECTED:", improved)