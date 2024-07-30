#!/bin/bash
#SBATCH --job-name=vessel_to_grid
#SBATCH --output=./output/test_%j.out
#SBATCH --error=./output/test_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=vascunet

PROJECT_DIR=$HOME/workspace/vascunet

source $PROJECT_DIR/.venv/bin/activate

python -u $PROJECT_DIR/src/utils/tester.py 