#!/bin/bash --login
#SBATCH -p gpuV               # Partition: A100 GPU / V100 GPU
#SBATCH --gres=gpu:2          # Request 2 GPUs
#SBATCH --time=3-00:00:00     # Wall time: 3 days
#SBATCH --ntasks=1            # One task
#SBATCH --cpus-per-task=16    # 16 CPU cores

# Load CUDA module
module purge
module load libs/cuda

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_CPUS_PER_TASK CPU core(s)"

# Activate Conda environment
conda activate top_analysis

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Run Optuna optimization
echo "Running hyperparameter optimization..."
python optuna_tuning.py

echo "Optimization completed at: $(date)"