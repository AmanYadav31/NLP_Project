import torch
from collections import Counter
def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss))

def calculate_distinct_n(tokens, n):
    total_ngrams = len(tokens) - n + 1
    ngrams = [tuple(tokens[i:i + n]) for i in range(total_ngrams)]
    unique_ngrams = len(Counter(ngrams))
    distinct_n = unique_ngrams / total_ngrams
    return distinct_n
