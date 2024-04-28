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

## Technical Documentation ##

1. **[vLLM](https://docs.vllm.ai/en/latest/index.html)**: For high-throughput low-latency inference via LLMs.
