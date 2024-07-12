#!/bin/bash
#SBATCH --job-name=vessel_to_grid
#SBATCH --output=./output/vtg_%j.out
#SBATCH --error=./output/vtg_%j.err
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=vis

PROJECT_DIR=$HOME/workspace/vascunet

source $PROJECT_DIR/.venv/bin/activate

python -u $PROJECT_DIR/src/utils/vessel_to_grid.py 