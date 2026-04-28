from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_answer(query, passages):
    context = "\n".join([f"[{pid}] {text}" for pid, text in passages])

    prompt = f"""
Answer the question using ONLY the passages.
Include citations like [P0].

If not enough information, say: Not enough information.

Question: {query}

Passages:
{context}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)