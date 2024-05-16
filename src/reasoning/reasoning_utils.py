import datasets
import numpy as np
import re
from typing import List, Tuple

from reasoning_configs import DATASETS_CONFIGURATION, EXEMPLARS

def load_dataset(dataset_name: str) -> datasets.Dataset:
    """Load the dataset based on the dataset name

    Args:
        dataset_name (str): Name of the dataset

    Raises:
        ValueError: Dataset not found

    Returns:
        datasets.Dataset: The loaded dataset
    """
    # Check if the dataset is present in the configuration
    if dataset_name not in DATASETS_CONFIGURATION:
        raise ValueError(f"Dataset {dataset_name} not found.")
    # Load the dataset based on the configuration
    dataset_config = DATASETS_CONFIGURATION[dataset_name]
    # Load the datasets based on the mentioned splits
    split_datasets = [
        datasets.load_dataset(
            path=dataset_config["path"], 
            name=dataset_config["name"], 
            split=split
        ) for split in dataset_config["split"]
    ]
    # Concatenate the datasets and return
    return datasets.concatenate_datasets(split_datasets)

def convert_dataset_to_prompts_and_answers(dataset: datasets.Dataset, dataset_name: str, instruction_tuned: bool, num_shots: int) -> Tuple[List[str]]:
    """Convert the dataset to prompts and answers

    Args:
        dataset (datasets.Dataset): The dataset to convert
        dataset_name (str): Name of the dataset
        instruction_tuned (bool): Flag to use instruction-tuned models
        num_shots (int): Number of shots to use for inference

    Returns:
        Tuple[List[str]]: The prompts and answers for the dataset
    """
    # Formulate the any-shot prefix
    shots_prefix = ""
    for shot in EXEMPLARS[dataset_name][:num_shots]:
        shots_prefix += f"Q: {shot['question']}\nA: {shot['answer']}\n\n"
    # Iterate over the dataset and convert each instance to a prompt and answer
    prompts, answers = [], []
    for qa_pair in dataset:
        prompts.append(
            _formulate_prompt(
                qa_pair[DATASETS_CONFIGURATION[dataset_name]['question']], 
                instruction_tuned, 
                DATASETS_CONFIGURATION[dataset_name]['instruction'],
                shots_prefix
            )
        )
        answers.append(qa_pair[DATASETS_CONFIGURATION[dataset_name]['answer']])
    return prompts, answers

def get_answer_distribution(sampled_sequences: List[str], dataset_name: str) -> List[Tuple[str, int]]:
    """Get the answer distribution from the sampled sequences
    
    Args:
        sampled_sequences (List[str]): Sampled sequences from the model
        dataset_name (str): Name of the dataset
        
    Returns:
        List[Tuple[str, int]]: The answer distribution
    """
    # Clean the answers and remove empty strings
    answers = [_answer_cleaning(sequence, dataset_name) for sequence in sampled_sequences]
    # No longer needed
    # answers = [answer.strip() for answer in answers if answer != ""]
    # If no answers are present, return an empty list
    if len(answers) == 0:
        return []
    # Get the answer distribution
    answers, answer_counts = np.unique(answers, return_counts=True)
    distribution = list(zip([str(answer) for answer in answers], answer_counts))
    return sorted(distribution, key=lambda x: x[1], reverse=True)

def get_majority_vote_answer(sampled_sequences: List[str], dataset_name: str) -> str:
    """Get the majority vote answer from the sampled sequences

    Args:
        sampled_sequences (List[str]): Sampled sequences from the model
        dataset_name (str): Name of the dataset

    Returns:
        str: The majority vote answer
    """
    
    answer_distribution = get_answer_distribution(sampled_sequences, dataset_name)
    
    # If no answers are present, return an empty string
    if len(answer_distribution) == 0:
        return ""
    
    return answer_distribution[0][0]

def _formulate_prompt(question: str, instruction_tuned: bool, instruction_prefix: str, shots_prefix: str) -> str:
    """Formulate a prompt for the reasoning task

    Args:
        question (str): The question to be answered
        instruction_tuned (bool): Flag to use instruction-tuned models
        shots_prefix (str): The prefix for the any-shot learning

    Raises:
        NotImplementedError: Instruction-tuned models are not yet supported, as they require special instruction tokens.

    Returns:
        str: The formulated prompt
    """
    prompt = ""
    # Add the instruction prefix with special tokens if instruction-tuned else add the instruction prefix as is
    if instruction_tuned:
        raise NotImplementedError("Instruction-tuned models are not yet supported.")
    else:
        prompt += f"{instruction_prefix}\n\n"
    # Add the shots prefix and the target question
    prompt += shots_prefix + f"Q: {question}\nA:"
    return prompt

def _answer_cleaning(sequence: str, dataset_name: str) -> str:
    """Extract the answer from the generated sequence

    Args:
        sequence (str): Sequence generated by the model
        dataset_name (str): Name of the dataset

    Returns:
        str: The extracted answer
    """
    is_mathematical = DATASETS_CONFIGURATION[dataset_name]['is_mathematical']
    # Extract the answer from the sequence based on the dataset type
    if is_mathematical:
        return _mathematical_answer_cleaning(sequence)
    else:
        return sequence.strip().split()[-1]

def _mathematical_answer_cleaning(sequence: str) -> str:
    """Extract the mathematical answer from the generated sequence

    Args:
        sequence (str): Sequence generated by the model

    Returns:
        str: The extracted mathematical answer
    """
    # Extract the numerical answer from the sequence
    answer = [s for s in re.findall(r'-?\d+\.?\d*', sequence.replace(",", ""))]
    # Return the answer if present, else return an empty string
    if len(answer) == 0:
        return -np.inf
    # Handle the case where the answer ends with a period
    return float(answer[-1][:-1] if answer[-1].endswith(".") else answer[-1])