from nltk import ngrams
import numpy as np
import json
import random
from typing import List, Tuple
def compute_n_gram_diversity(qa_generations: List[object], subsample_size: int | None = None) -> float:
    """Compute the n-gram diversity for the QA generation output

    Args:
        qa_generations (List[object]): List of question-answer generation outputs

    Returns:
        float: n-gram diversity for 1-4 grams
    """
    n_gram_diversity = 0.0
    # Compute the n-gram diversity for 1-4 grams
    for qa in qa_generations:
        # Subsample the generated sequences
        sentences_before_sampling = qa['generated_sequences']
        sentences = random.sample(sentences_before_sampling, subsample_size)
        # Compute the n-gram diversity for the QA generation
        qa_n_gram_diversity = distinct_n_gram_helper(sentences)
        n_gram_diversity += qa_n_gram_diversity
    # Return the n-gram diversity
    return n_gram_diversity / len(qa_generations)

def distinct_n_gram_helper(sentences: List[str]):
    """Compute distinct-n for n in [1,4] a list of sentences
    Args:
        sentences (List[str]): a list of sentences
    
    Returns:
        float: distinct-n score for n in [1,4]
    """
    # Initialize the distinct-n score
    distinct_n = 0.0
    # Compute the distinct-n score for n in [1,4]
    for n in range(1, 5):
        corpus_n_grams = [*[ngrams(sentence, n) for sentence in sentences]]
        distinct_n_grams = set(corpus_n_grams)

        distinct_n += len(distinct_n_grams) / (len(corpus_n_grams) + np.finfo(float).eps)
    # Return the distinct-n score
    return distinct_n / 4


#BASELINE GSM8K GEMMA2-B
with open('/users/poojithapenta/desktop/COMPSCI-685-PROJECT/src/Plots/baseline__gsm8k__gemma-2b__output.json', 'r') as file:
    baseline_gemma2b_data = json.load(file)

#BASELINE GSM8K GEMMA 7-B

with open('/users/poojithapenta/desktop/COMPSCI-685-PROJECT/src/Plots/baseline__gsm8k__gemma-7b__output.json', 'r') as file:
    baseline_gemma7b_data = json.load(file)

# BASELINE GSMK GPT2-LARGE
with open('/users/poojithapenta/desktop/COMPSCI-685-PROJECT/src/Plots/baseline__gsm8k__gpt2-large__output.json', 'r') as file:
    baseline_gpt2_data = json.load(file)


def n_gram_diversity(data):
    samples=[5,10,15,20,25,30,35,40]
    ngramm=[]
    for i in samples:
        x=compute_n_gram_diversity(data, i)
        ngramm.append(x)

    return ngramm
y=n_gram_diversity(baseline_gemma7b_data)
print(y)