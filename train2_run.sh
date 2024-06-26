#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=./output/train2_%j.out
#SBATCH --error=./output/train2_%j.err
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=vis
#SBATCH --gres=gpu:1

PROJECT_DIR=$HOME/workspace/vascunet

source $PROJECT_DIR/.venv/bin/activate

python $PROJECT_DIR/src/train2.py --seed 666 --batch_size 2 --learning_rate 1e-3 --num_points 1024 --num_epochs 30 --model VesselModelSinglePoint2 --dataset VesselDatasetSinglePoint
 