#!/bin/bash
#SBATCH -c 4                        # Number of Cores per Task
#SBATCH --mem=50G                   # Requested Memory
#SBATCH -p gpu                      # Partition
#SBATCH -G 1                        # Number of GPUs
#SBATCH --constraint vram11         # GPU Memory
#SBATCH -t 04:00:00                 # Job time limit
#SBATCH -o job-%j-logs.out          # %j = job ID

module load miniconda/22.11.1-1
conda activate cs685

python3 src/reasoning/main.py --model "openai-community/gpt2-large" --dataset "gsm8k" --dataset-size 8792 \
    --recurring-self-consistency \
    --num-shots 8 \
    --max-out-tokens 120 \
    --gpu-memory-utilization 0.7 \
    --output-file "results/recurring__gsm8k__gpt2-large__output.json"