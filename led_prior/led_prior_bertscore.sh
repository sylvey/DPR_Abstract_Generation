#!/bin/bash

# sbatch -N 1 -p GPU-shared -t 40:00:00 --gpus=v100-32:1 -A cis230089p upgrade.sh
# squeue -u $USER


# echo commands to stdout - these are saved to a slurm log file in the same directory as your sh file
set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change directory to wherever your python file you are trying to run
# Example: cd /ocean/projects/cis230089p/jmenke/multitagger_v2
cd /ocean/projects/cis230089p/slin23/full_text_label/led_prior

# activate conda environment -  you have to do this through the source command - you'll also have to download miniconda and set up an environment in your home directory
# Example: source /jet/home/jmenke/miniconda3/bin/activate multitagger
source /ocean/projects/cis230089p/slin23/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis230089p/slin23/miniconda3/envs/led_fp16


# pip install datasets==1.2.1
# pip install transformers==4.2.0
# pip install rouge_score
# pip install dill==0.3.6

# Input python file name you want to run along with arguments 
# Example: python train.py --train_val_test="train" --max_epoch=25
# echo "PYTHON PATH:"
# which python

# echo "CHECK TORCH:"
# python -c "import torch; print(torch.__version__)"



python bertscore.py models/preds/test_generations_20251008_034111.jsonl \
  --model_type ../led/models/scibert --filter_errors --batch_size 16 --device cuda
