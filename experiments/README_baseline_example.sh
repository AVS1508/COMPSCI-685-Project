#!/bin/bash
#SBATCH -c 4                        # Number of Cores per Task
#SBATCH --mem=8G                    # Requested Memory
#SBATCH -p gpu                      # Partition
#SBATCH -G 1                        # Number of GPUs
#SBATCH --constraint vram11         # GPU Memory
#SBATCH -t 00:30:00                 # Job time limit
#SBATCH -o job-%j-logs.out          # %j = job ID

python3 src/reasoning/main.py --model "facebook/opt-125m" --dataset "gsm8k" --dataset-size 8792 \
    --num-samples 5 \
    --num-shots 8 \
    --max-out-tokens 120 \
    --gpu-memory-utilization 0.7 \
    --output-file "results/baseline_example_output.json"