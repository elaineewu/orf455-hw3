#!/bin/bash
#SBATCH --job-name=q7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=YOUR_NETID@princeton.edu

# Runs every Q7 scenario on Adroit, then makes the tables and figures.
# Submit from ~/orf455-q7 with:   sbatch run_q7.slurm
# Check on it with:               squeue -u $USER
# Printed output goes to slurm-<jobid>.out in the same folder.

module purge
module load anaconda3/2022.10 gurobi
conda activate vatic-test

cd ~/orf455-q7
python q7.py a --solver gurobi --threads 4
python q7.py b --solver gurobi --threads 4
python q7.py c --solver gurobi --threads 4 --seeds 5
python q7.py summarize
