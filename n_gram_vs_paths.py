from nltk import ngrams
import numpy as np
import json
import random
from typing import List, Tuple
import matplotlib.pyplot as plt


def compute_n_gram_diversity(qa_generations: List[object], subsample_size: int = -1) -> float:
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
    total_unique_ngrams = 0
    n_gram_diversity = 0.0
    # Compute the distinct-n score for n in [1,4]
    for n in range(1, 5):
        unique_ngrams = set()
        total_ngrams_count = 0
        for sequence in sentences:
            n_grams = list(ngrams(sequence.split(), n))
            unique_ngrams.update(n_grams)
            total_ngrams_count += len(n_grams)
        total_unique_ngrams = len(unique_ngrams)
        n_gram_diversity += total_unique_ngrams / (total_ngrams_count + np.finfo(float).eps)
    return n_gram_diversity


# BASELINE GSM8K GEMMA2-B
with open('results/baseline__gsm8k__gemma-2b__output.json', 'r') as file:
    baseline_gemma2b_data = json.load(file)
# BASELINE GSM8K GEMMA 7-B
with open('results/baseline__gsm8k__gemma-7b__output.json', 'r') as file:
    baseline_gemma7b_data = json.load(file)
# BASELINE GSMK GPT2-LARGE
with open('results/baseline__gsm8k__gpt2-large__output.json', 'r') as file:
    baseline_gpt2_data = json.load(file)
# GREEDY GEMMA2-B
with open('results/greedy__gsm8k__gemma-2b__output.json', 'r') as file:
    greedy_gemma2b_data = json.load(file)
# GREEDY GEMMA 7-B
with open('results/greedy__gsm8k__gemma-7b__output.json', 'r') as file:
    greedy_gemma7b_data = json.load(file)
# GREEDY GPT2-LARGE
with open('results/greedy__gsm8k__gpt2-large__output.json', 'r') as file:
    greedy_gpt2_data = json.load(file)
# RECURRING GSM8K GEMMA2-B
with open('results/recurring__gsm8k__gemma-2b__output.json', 'r') as file:
    recurring_gemma2b_data = json.load(file)
# RECURRING GSM8K GEMMA 7-B
with open('results/recurring__gsm8k__gemma-7b__output.json', 'r') as file:
    recurring_gemma7b_data = json.load(file)
# RECURRING GSMK GPT2-LARGE
with open('results/recurring__gsm8k__gpt2-large__output.json', 'r') as file:
    recurring_gpt2_data = json.load(file)

def n_gram_diversity(data, is_greedy=False):
    samples=[5,10,15,20,25,30,35,40]
    ngramm=[]
    for i in samples:
        x = compute_n_gram_diversity(data, i if not is_greedy else 1)
        ngramm.append(x)
    return ngramm

def n_gram_diversity_recur(data):
    samples=[8, 16, 24, 32, 40]
    ngramm=[]
    for idx, sample in enumerate(samples):
        n_gram_diversity = np.sum([distinct_n_gram_helper(instance['generated_sequences'][idx] if idx < len(instance['generated_sequences']) else [])  for instance in data]) / len(data)
        ngramm.append(n_gram_diversity)
    return ngramm

# y = n_gram_diversity(baseline_gpt2_data)
# print(y)
plt.plot([5,10,15,20,25,30,35,40], n_gram_diversity(greedy_gemma2b_data, is_greedy=True), marker='o', label='greedy')
plt.plot([5,10,15,20,25,30,35,40], n_gram_diversity(baseline_gemma2b_data), marker='o', label='baseline')
plt.plot([8, 16, 24, 32, 40], n_gram_diversity_recur(recurring_gemma2b_data), marker='o', label='recurring')
plt.title('Variation of n-gram diversity with number of sampled reasoning paths')
plt.xlabel('Number of sampled reasoning paths')
plt.ylabel('N-gram diversity')
plt.legend()
plt.savefig('ngram_diversity_gemma-2b.png')
plt.close()

plt.plot([5,10,15,20,25,30,35,40], n_gram_diversity(greedy_gemma7b_data, is_greedy=True), marker='o', label='greedy')
plt.plot([5,10,15,20,25,30,35,40], n_gram_diversity(baseline_gemma7b_data), marker='o', label='baseline')
plt.plot([8, 16, 24, 32, 40], n_gram_diversity_recur(recurring_gemma7b_data), marker='o', label='recurring')
plt.title('Variation of n-gram diversity with number of sampled reasoning paths')
plt.xlabel('Number of sampled reasoning paths')
plt.ylabel('N-gram diversity')
plt.legend()
plt.savefig('ngram_diversity_gemma-7b.png')
plt.close()

plt.plot([5,10,15,20,25,30,35,40], n_gram_diversity(greedy_gpt2_data, is_greedy=True), marker='o', label='greedy')
plt.plot([5,10,15,20,25,30,35,40], n_gram_diversity(baseline_gpt2_data), marker='o', label='baseline')
plt.plot([8, 16, 24, 32, 40], n_gram_diversity_recur(recurring_gpt2_data), marker='o', label='recurring')
plt.title('Variation of n-gram diversity with number of sampled reasoning paths')
plt.xlabel('Number of sampled reasoning paths')
plt.ylabel('N-gram diversity')
plt.legend()
plt.savefig('ngram_diversity_gpt2-large.png')
plt.close()