#!/bin/bash
#SBATCH -c 4                        # Number of Cores per Task
#SBATCH --mem=50G                   # Requested Memory
#SBATCH -p gpu                      # Partition
#SBATCH -G 1                        # Number of GPUs
#SBATCH --constraint vram48         # GPU Memory
#SBATCH -t 12:00:00                 # Job time limit
#SBATCH -o job-%j-logs.out          # %j = job ID

module load miniconda/22.11.1-1
conda activate cs685

python3 src/reasoning/main.py --model "google/gemma-7b" --dataset "gsm8k" --dataset-size 8792 \
    --recurring-self-consistency \
    --num-shots 8 \
    --max-out-tokens 120 \
    --gpu-memory-utilization 0.9 \
    --output-file "results/recurring__gsm8k__gemma-7b__output-optimized.json"
