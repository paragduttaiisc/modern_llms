#!/bin/bash
#SBATCH --job-name=gpt2_muon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --nodelist=n2
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs
mkdir -p outputs

module load miniconda
source activate llm

export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    train.py \
    --batch-size=128 \
    --n-iters=19000 \
    --last-decay-iter=17000 \
    --save-dir="outputs" \
    --use-bf16