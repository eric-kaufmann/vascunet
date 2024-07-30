#!/bin/bash
#SBATCH --job-name=autoencoder128
#SBATCH --output=./output/autoencoder128_%j.out
#SBATCH --error=./output/autoencoder128_%j.err
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=vascunet
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

PROJECT_DIR=$HOME/workspace/vascunet

source $PROJECT_DIR/.venv/bin/activate

python $PROJECT_DIR/src/train_autoencoder128.py --learning_rate 0.00001 --job_id $SLURM_JOB_ID --num_epochs 300 --batch_size 1 --log_every 400 --vector_input --add_metadata