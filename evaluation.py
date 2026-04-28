def is_correct(answers, text):
    return any(ans.lower() in text.lower() for ans in answers)

def recall_at_k(results, answers, k):
    return int(any(is_correct(answers, doc) for _, doc in results[:k]))

def precision_at_k(results, answers, k):
    correct = sum(is_correct(answers, doc) for _, doc in results[:k])
    return correct / k

def mrr(results, answers):
    for i, (_, doc) in enumerate(results):
        if is_correct(answers, doc):
            return 1 / (i + 1)
    return 0