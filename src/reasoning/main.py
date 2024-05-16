from arguments import ReasoningArgumentsParser
from reasoning_llm import ReasoningLLM

# BEGIN - Example Usage of the Self-Consistency CoT Reasoning Module #
# 1. Example usage of the Greedy CoT Reasoning Module

# # python src/reasoning/main.py --model facebook/opt-125m --output_file greedy-gsm8k.json \
# # --dataset gsm8k --greedy --dataset-seed 1379 --generation-seed 42

# 2. Example usage of the Baseline Self-Consistency CoT Reasoning Module

# # python src/reasoning/main.py --model facebook/opt-125m --output_file self-consistency-baseline-gsm8k.json \
# # --dataset gsm8k --num_samples 10 --temperature 0.5 --top_k 40 --dataset-seed 1379 --generation-seed 42

# 3. Example usage of the Recurring Self-Consistency CoT Reasoning Module

# # python src/reasoning/main.py --model facebook/opt-125m --output_file self-consistency-recurring-gsm8k.json \
# # --dataset gsm8k --recurring-self-consistency --num_samples 10 --temperature 0.5 --top_k 40 \
# # --majority-threshold 0.5 --time-steps 5 --dataset-seed 1379 --generation-seed 42

# END - Example Usage of the Self-Consistency CoT Reasoning Module #


if __name__ == '__main__':
    # Parse the command-line arguments as keyword arguments
    reasoning_kwargs = ReasoningArgumentsParser().keyword_arguments()
    # Initialize the reasoning module with the parsed keyword arguments
    llm = ReasoningLLM(**reasoning_kwargs)
    # Run the reasoning experiment based on the specified pipeline
    llm.run_experiment()