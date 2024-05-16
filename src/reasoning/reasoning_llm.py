import json
from typing import List
from vllm import LLM, SamplingParams
from reasoning_utils import (
    load_dataset,
    convert_dataset_to_prompts_and_answers,
    get_majority_vote_answer,
    get_answer_distribution,
)

class ReasoningLLM:
    def __init__(self, **kwargs):
        """Module for the chain-of-thought reasoning tasks via LLM inference.
        """
        # Initialize the LLM model with the specified model and GPU memory utilization
        self.model = LLM(
            model = kwargs['model'], 
            gpu_memory_utilization = kwargs['gpu_memory_utilization'],
            dtype="half"
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
        
        # Loads extra parameters for the recurring self-consistency pipeline
        if self.recurring_self_consistency:
            self.use_majority_threshold = kwargs['use_majority_threshold']
            self.majority_threshold = kwargs['majority_threshold']
            self.num_samples_per_time_step = kwargs['samples_per_time_step']
            self.time_steps = kwargs['time_steps']
    
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
        """
        if not self.recurring_self_consistency:
            raise ValueError("Recurrent self-consistency pipeline not selected")
        
        
        outputs = [{
            "input_prompts": [],
            "ground_truth_reasonings": self.answers[i].split("####")[0],
            "ground_truth_answers": self.answers[i].split("####")[1].strip(),
            "generated_sequences": [],
            "answer_distribution": [],
            "majority_vote_answers": [],
        } for i in range(len(self.prompts))]
        
        for time_step in range(1, self.time_steps+1):
            generations = self.model.generate(
                self.prompts,
                self.sampling_params
            )
            
            for generation_index, generation in enumerate(generations):
                input_prompt = generation.prompt
                generated_sequences = [generation.outputs[i].text for i in range(len(generation.outputs))]
                answer_distribution = get_answer_distribution(generated_sequences, self.dataset_name)
                majority_vote_answer = str(answer_distribution[0][0]) if len(answer_distribution) > 0 else ""
                
                outputs[generation_index]["input_prompts"].append(input_prompt)
                outputs[generation_index]["generated_sequences"].append(generated_sequences)
                outputs[generation_index]["answer_distribution"].append(answer_distribution)
                outputs[generation_index]["majority_vote_answers"].append(majority_vote_answer)