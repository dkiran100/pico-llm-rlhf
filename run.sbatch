#!/bin/bash

#SBATCH --job-name=transformer_sweep
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --account=pr_95_tandon_priority
#SBATCH --output=transformer_sweep.log
#SBATCH --error=transformer_sweep_err.log
singularity exec \
--nv --overlay /scratch/ar8692/filesystems/overlay-25GB-500K.ext3:ro \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;
conda activate picollm_env;
cd /scratch/ar8692/pico-llm/src/;
wandb agent sus221-new-york-university/pico-llm-sweeps/p9gbsvw3
"