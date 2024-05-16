#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=./output/train_%j.out
#SBATCH --error=./output/train_%j.err
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mail-user=eric.kaufmann@uni-jena.de
#SBATCH --mail-type=ALL

PROJECT_DIR=$HOME/workspace/vascunet

# Prepare personal directory on the node's scratch partition
SCRATCH_DIR=/scratch/$USER/vascunet
mkdir -p $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/output/

# Copy input data to the node's scratch partition
cp -r $PROJECT_DIR/ $SCRATCH_DIR/

# Activate your virtual environment if needed
source $SCRATCH_DIR/.venv/bin/activate

# Run the Python script
python $SCRATCH_DIR/src/train.py

# Copy output data from the node's scratch partition and clean up
cp -r $SCRATCH_DIR/lightning_logs/ $PROJECT_DIR/lightning_logs/
cp -r $SCRATCH_DIR/output/ $PROJECT_DIR/output/

rm -rf $SCRATCH_DIR/