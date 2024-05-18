import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from nltk import ngrams

n, k = 8792, 5

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def plot_mean_unique_answers_vs_time_steps():
    mean_unique_counts_gpt2_large = calculate_num_unique_answer_vs_time_steps("results/recurring__gsm8k__gpt2-large__output.json")
    mean_unique_counts_gemma_2b = calculate_num_unique_answer_vs_time_steps("results/recurring__gsm8k__gemma-2b__output.json")
    
    x_values = [i + 1 for i in range(5)]
    y_values = [mean_unique_counts_gpt2_large, mean_unique_counts_gemma_2b]
    labels = ["GPT-2 Large", "Gemma-2B"]
    
    plot_graph(x_values, y_values, labels, 'Time Step', 'Mean Number of Unique Answers', 'Mean Number of Unique Answers v/s Time Step', "results/mean_unique_answers_vs_time_steps.png")
    

def calculate_num_unique_answer_vs_time_steps(file_path):
    data = read_json_file(file_path)
        
    # Extract the number of unique answers per time step
    unique_counts = np.zeros((n, k))

    for i, instance in enumerate(data):
        for j, timestep in enumerate(instance["answer_distribution"]):
            unique_counts[i, j] = len(timestep)
            
    # Calculate the mean number of unique answers per time step
    # mean_unique_counts = np.sum(unique_counts, axis=0)
    mean_unique_counts = np.mean(unique_counts, axis=0)
    return mean_unique_counts

def compute_n_gram_diversity(qa_generations: List[str]):
    """Compute the n-gram diversity for the QA generation output

    Args:
        qa_generations (List[object]): List of question-answer generation outputs

    Returns:
        float: n-gram diversity for 1-4 grams
    """
    n_gram_diversity = 0.0
    max_ngram_diversity = 0.0
    # Compute the n-gram diversity for 1-4 grams
    for qa in qa_generations:
        # Subsample the generated sequences
        sentence = qa
        # Compute the n-gram diversity for the QA generation
        qa_n_gram_diversity = distinct_n_gram_helper([sentence])
        if qa_n_gram_diversity > max_ngram_diversity:
            max_ngram_diversity = qa_n_gram_diversity
        n_gram_diversity += qa_n_gram_diversity
    # Return the n-gram diversity
    mean_ngram_diversity = n_gram_diversity / len(qa_generations)
    
    return [mean_ngram_diversity, max_ngram_diversity]
    

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

def plot_mean_ngram_diversity_vs_time_steps():
    gpt_2_large_fp = "results/recurring__gsm8k__gpt2-large__output.json"
    gemma_2b_fp = "results/recurring__gsm8k__gemma-2b__output.json"
    
    gpt_2_large_data = read_json_file(gpt_2_large_fp)
    gpt_2_large_mean_ngram_diversity = np.zeros((n, k))
    gpt_2_large_max_ngram_diversity = np.zeros((n, k))
    
    for i, instance in enumerate(gpt_2_large_data):
        for j, sequences in enumerate(instance["generated_sequences"]):
            instance_mean_ngram_diversity, instance_max_ngram_diversity = compute_n_gram_diversity(sequences)
            gpt_2_large_mean_ngram_diversity[i, j] = instance_mean_ngram_diversity
            gpt_2_large_max_ngram_diversity[i, j] = instance_max_ngram_diversity
            
    gpt_2_large_mean_ngram_diversity = np.mean(gpt_2_large_mean_ngram_diversity, axis=0).reshape(1, -1)
    gpt_2_large_max_ngram_diversity = np.mean(gpt_2_large_max_ngram_diversity, axis=0).reshape(1, -1)
    
    
    gemma_2b_data = read_json_file(gemma_2b_fp)
    gemma_2b_data_mean_ngram_diversity = np.zeros((n, k))
    gemma_2b_data_max_ngram_diversity = np.zeros((n, k))
    
    for i, instance in enumerate(gemma_2b_data):
        for j, sequences in enumerate(instance["generated_sequences"]):
            instance_mean_ngram_diversity, instance_max_ngram_diversity = compute_n_gram_diversity(sequences)
            gemma_2b_data_mean_ngram_diversity[i, j] = instance_mean_ngram_diversity
            gemma_2b_data_max_ngram_diversity[i, j] = instance_max_ngram_diversity
            
    gemma_2b_data_mean_ngram_diversity = np.mean(gemma_2b_data_mean_ngram_diversity, axis=0).reshape(1, -1)
    gemma_2b_data_max_ngram_diversity = np.mean(gemma_2b_data_max_ngram_diversity, axis=0).reshape(1, -1)
    
    print(gemma_2b_data_mean_ngram_diversity.shape)
    print(gemma_2b_data_mean_ngram_diversity)
    print(gemma_2b_data_max_ngram_diversity.shape)
    
    x_values = np.array([i + 1 for i in range(5)]).reshape(1, -1)
    
    plot_graph(x_values, [gpt_2_large_mean_ngram_diversity, gemma_2b_data_mean_ngram_diversity], ["GPT-2 Large", "Gemma-2B"], 'Time Step', 'Mean N-gram Diversity', 'Mean N-gram Diversity v/s Time Step', "results/mean_ngram_diversity_vs_time_steps.png")
    
    plot_graph(x_values, [gpt_2_large_max_ngram_diversity, gemma_2b_data_max_ngram_diversity], ["GPT-2 Large", "Gemma-2B"], 'Time Step', 'Max N-gram Diversity', 'Max N-gram Diversity v/s Time Step', "results/max_ngram_diversity_vs_time_steps.png")
    

def plot_graph(x, y, labels, xlabel, ylabel, title, file_path):
    for i in range(len(y)):
        plt.plot(x, y[i], marker='o', label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(file_path)
    plt.clf()
    plt.cla()

if __name__ == '__main__':
    # plot_mean_unique_answers_vs_time_steps()
    plot_mean_ngram_diversity_vs_time_steps()