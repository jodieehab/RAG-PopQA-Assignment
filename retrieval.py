from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(texts):
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, texts

def search(query, index, texts, k=5):
    q_emb = model.encode([query])
    _, ids = index.search(np.array(q_emb), k)

    return [(f"P{i}", texts[i][:200]) for i in ids[0]]