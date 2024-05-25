#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=./output/train_%j.out
#SBATCH --error=./output/train_%j.err
#SBATCH --ntasks=1
#SBATCH --time=2:30:00
#SBATCH --mail-user=eric.kaufmann@uni-jena.de
#SBATCH --mail-type=ALL
#SBATCH --partition=vis
#SBATCH --gres=gpu:1

PROJECT_DIR=$HOME/workspace/vascunet

source $PROJECT_DIR/.venv/bin/activate

python $PROJECT_DIR/src/train.py