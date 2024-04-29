# COMPSCI 685 Project #

**Title:** Lower Sampling with Recurring Elimination Improves Self-Consistency for Chain of Thought Reasoning

**Team:** Aadam Lokhandwala, Aditya Vikram Singh, Dhrumeen Patel, Poojitha Penta, Sahil Gupta

## Installation and Setting Up ##

```bash
# Setup a conda environment
conda create -n cs685 python=3.9 -y
conda activate cs685

# Install the required packages
pip install -r requirements.txt

# Run the vLLM test file
python3 src/generation/vllm_test.py
```

## Example of Experimentation - Baseline Self-Consistency ##

1. Run the example experiment in `./experiments` directory

    ```bash
    # Activate the conda environment
    conda activate cs685
    # Run the baseline example experiment
    bash experiments/baseline_example.sh > logs/baseline_experiments_logs.out
    ```

2. The results will be stored in the `./results/baseline_example_output.json` file and logs will be stored in the `./logs` directory.

3. Review the command in the `baseline_example.sh` file to understand the parameters used for the experiment.

## Technical Documentation ##

### Folder Structure ###

1. **`src/`**: Contains the Python source code for the project.
    - **`reasoning/`**: Contains the code for reasoning and generating the texts.
    - **`metrics/`**: (TBD) Contains the code for evaluating the generated texts.
    - **`testing/`**: Contains the code for testing the LLMs inference with vLLM and Flash Attention.
2. **`experiments/`**: Contains the scripts for running the experiments as bash files.
3. **`results/`**: Contains the results of the experiments as JSON files.
4. **`experiment_logs/`**: Contains the logs of the experiments as text files.

### Dependencies ###

1. **[vLLM](https://docs.vllm.ai/en/latest/index.html)**: For high-throughput low-latency inference via LLMs, limited support for some LLMs.
2. **[flash-attn](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention)**: For fast and efficient inference with LLMs using Flash Attention.  
