import argparse
from typing import Dict

class ReasoningArgumentsParser:
    def __init__(self) -> None:
        """Argument parser for the reasoning module
        """
        # Initialize the argument parser
        self.parser = argparse.ArgumentParser(description='Generate sequences via LLM inference')
        # Add arguments pertaining to the inference inputs and outputs for model
        self.parser.add_argument('--model', type=str, default='facebook/opt-125m', help='LLM to use for inference')
        self.parser.add_argument('--instruction-tuned', action='store_true', help='Flag to use instruction-tuned models')
        self.parser.add_argument('--num-shots', type=int, default=0, help='Number of shots to use for inference')
        self.parser.add_argument('--output-file', type=str, default='results/output.json', help='JSON file to write the generated sequences and information to')
        # Add arguments pertaining to the inference inputs and outputs for dataset
        self.parser.add_argument('--dataset', type=str, default='gsm8k', help='Dataset to perform inference on')
        self.parser.add_argument('--dataset-size', type=int, default=100, help='Number of dataset instances to use for inference')
        self.parser.add_argument('--dataset-offset', type=int, default=0, help='Offset for dataset instances to use for inference')
        # Add arguments pertaining to the GPU memory utilization and maximum output tokens
        self.parser.add_argument('--gpu-memory-utilization', type=float, default=0.8, help='Fraction of GPU memory to use')
        self.parser.add_argument('--max-out-tokens', type=int, default=512, help='Maximum number of tokens to generate for each sequence')
        # Add arguments for sampling parameters/knobs
        self.parser.add_argument('--num-samples', '-N', type=int, default=5, help='Number of samples to generate for each prompt')
        self.parser.add_argument('--temperature', '-T', type=float, default=0.5, help='Temperature knob for sampling')
        self.parser.add_argument('--top-k', '-k', type=int, default=40, help='Top-k knob for sampling')
        self.parser.add_argument('--top-p', '-p', type=float, default=1, help='Top-p (nucleus) knob for sampling')
        self.parser.add_argument('--stop', type=str, default='\n\n', help='Stop token for sampling')
        # Add argument for the greedy decoding and modified self-consistency pipeline, default is baseline self-consistency
        self.parser.add_argument('--greedy', action='store_true', help='Flag to use greedy decoding instead of sampling')
        self.parser.add_argument('--recurring-self-consistency', action='store_true', help='Flag to use the modified self-consistency pipeline instead of the baseline self-consistency')
        # Add arguments for the new pipeline
        self.parser.add_argument('--use-majority-threshold', '-M', action='store_true', help='Flag to use majority threshold for termination in the recurrent pipeline')
        self.parser.add_argument('--majority-threshold', '-m', type=float, default=0.5, help='Threshold for the majority consistency for termination in the recurrent pipeline')
        self.parser.add_argument('--samples-per-time-step', '-s', type=int, default=8, help='Number of samples to generate at each time step in the recurrent pipeline')
        self.parser.add_argument('--time-steps', '-t', type=int, default=5, help='Maximum number of time steps to run the recurrent pipeline for')
        # Add arguments for the randomization seeds
        self.parser.add_argument('--dataset-seed', type=int, default=1379, help='Seed for randomized shuffling of data')
        self.parser.add_argument('--generation-seed', type=int, default=42, help='Seed for randomized generation of sequences')
    
    def keyword_arguments(self) -> Dict:
        """Parse the arguments and return them as keyword arguments

        Returns:
            Dict: The parsed keyword arguments as a dictionary
        """
        args = self.parser.parse_args()
        return {
            'model': args.model,
            'instruction_tuned': args.instruction_tuned,
            'num_shots': args.num_shots,
            'dataset': args.dataset,
            'dataset_size': args.dataset_size,
            'dataset_offset': args.dataset_offset,
            'output_file': args.output_file,
            'gpu_memory_utilization': args.gpu_memory_utilization,
            'max_out_tokens': args.max_out_tokens,
            'num_samples': args.num_samples,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'stop': args.stop,
            'greedy': args.greedy,
            'recurring_self_consistency': args.recurring_self_consistency,
            'majority_threshold': args.majority_threshold,
            'use_majority_threshold': args.use_majority_threshold,
            'samples_per_time_step': args.samples_per_time_step,
            'time_steps': args.time_steps,
            'dataset_seed': args.dataset_seed,
            'generation_seed': args.generation_seed
        }