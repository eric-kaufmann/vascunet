#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=%jtest_job.out
#SBATCH --error=%jtest_job.err
#SBATCH --time=00:00:01
#SBATCH --ntasks=1
#SBATCH --mail-user=eric.kaufmann@uni-jena.de
#SBATCH --mail-type=ALL

SCRATCH_DIR=/scratch/$USER/test/
mkdir -p $SCRATCH_DIR
mkdir -p $SCRATCH_DIR/output/

PROJECT_DIR=$HOME/workspace/vascunet/

# Your commands here
echo "Hello, SLURM!"
touch $SCRATCH_DIR/output/test.txt

# Copy output data from the node's scratch partition and clean up
cp -r $SCRATCH_DIR $PROJECT_DIR/output/

rm -rf $SCRATCH_DIR