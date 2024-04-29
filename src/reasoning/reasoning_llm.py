import json
from typing import List
from vllm import LLM, SamplingParams
from reasoning_utils import load_dataset,\
    convert_dataset_to_prompts_and_answers,\
    get_majority_vote_answer

class ReasoningLLM:
    def __init__(self, **kwargs):
        """Module for the chain-of-thought reasoning tasks via LLM inference.
        """
        # Initialize the LLM model with the specified model and GPU memory utilization
        self.model = LLM(
            model = kwargs['model'], 
            gpu_memory_utilization = kwargs['gpu_memory_utilization']
        )
        # Set the sampling parameters based on the keyword arguments
        self.sampling_params = SamplingParams(
            n = 1,
            max_tokens=kwargs['max_out_tokens'],
            seed=kwargs['generation_seed'],
            temperature=1.0,
            stop=kwargs['stop']
        ) if kwargs['greedy'] == True else SamplingParams(
            n = kwargs['num_samples'],
            max_tokens=kwargs['max_out_tokens'],
            seed=kwargs['generation_seed'],
            temperature=kwargs['temperature'],
            top_k=kwargs['top_k'],
            top_p=kwargs['top_p'],
            stop=kwargs['stop']
        )
        # Load the dataset through dataset configuration and select the specified range after shuffling
        self.dataset_name = kwargs['dataset']
        self.dataset = load_dataset(self.dataset_name) \
            .shuffle(kwargs['dataset_seed']) \
            .select(
                range(
                    kwargs['dataset_offset'], 
                    kwargs['dataset_offset'] + kwargs['dataset_size']
                )
            )
        # Convert the dataset to prompts and answers
        self.prompts, self.answers = convert_dataset_to_prompts_and_answers(
            self.dataset,
            self.dataset_name,
            kwargs['instruction_tuned'],
            kwargs['num_shots']
        )
        # Set the output file for the reasoning results
        self.output_file = kwargs['output_file']
        self.recurring_self_consistency = kwargs['recurring_self_consistency']
    
    def run_experiment(self) -> None:
        """Run the reasoning experiment based on the specified pipeline
        """
        if self.recurring_self_consistency:
            self.cot_reasoning_self_consistency_recurrent()
        else:
            self.cot_reasoning_self_consistency_baseline()
    
    def cot_reasoning_self_consistency_baseline(self) -> None:
        """Baseline reasoning pipeline for the chain-of-thought reasoning tasks
        """
        # Generate the sequences for the prompts
        generations = self.model.generate(
            self.prompts, 
            self.sampling_params
        )
        outputs = []
        # Iterate over the generations and save the reasoning results
        for generation_index, generation in enumerate(generations):
            input_prompt = generation.prompt
            generated_sequences = [generation.outputs[i].text for i in range(len(generation.outputs))]
            outputs.append({
                "input_prompt": input_prompt,
                "ground_truth_reasoning": self.answers[generation_index].split("####")[0],
                "ground_truth_answer": self.answers[generation_index].split("####")[1].strip(),
                "generated_sequences": generated_sequences,
                "majority_vote_answer": get_majority_vote_answer(generated_sequences, self.dataset_name),
            })
        # Save the reasoning results to the output file
        with open(self.output_file, 'w') as f:
            json.dump(outputs, f, indent=4)
        print(f"Reasoning results saved to {self.output_file}")
    
    def cot_reasoning_self_consistency_recurrent(self) -> None:
        """Recurrent reasoning pipeline for the chain-of-thought reasoning tasks

        Raises:
            NotImplementedError: Recurrent reasoning pipeline is not implemented yet.
        """
        raise NotImplementedError("Recurrent reasoning pipeline is not implemented yet.")