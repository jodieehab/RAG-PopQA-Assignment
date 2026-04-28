from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, results):
    pairs = [[query, doc] for _, doc in results]
    scores = model.predict(pairs)

    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked]