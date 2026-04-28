from datasets import load_dataset

def load_popqa(n=100):
    data = load_dataset("akariasai/PopQA")["test"]
    return data.select(range(n))

def load_wikipedia(n=2000):
    # use a stable open dataset
    wiki = load_dataset("ag_news", split="train")

    texts = []
    for item in wiki:
        text = item["text"]
        if len(text) > 50:
            texts.append(text)

    return texts[:n]