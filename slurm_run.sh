#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=./output/train_%j.out
#SBATCH --error=./output/train_%j.err
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mail-user=eric.kaufmann@uni-jena.de
#SBATCH --mail-type=ALL
#SBATCH --partition=vis
#SBATCH --gres=gpu:1

PROJECT_DIR=$HOME/workspace/vascunet

source $PROJECT_DIR/.venv/bin/activate

python $PROJECT_DIR/src/train.py --num_random_mesh_iterations 1 --num_fluid_samples 20000 --num_meshpoints 8192 --seed 666 --batch_size 128 --learning_rate 1e-5 --train_val_split 0.8 --num_neighbours 32 --num_midpoints 32 --num_epochs 50
 