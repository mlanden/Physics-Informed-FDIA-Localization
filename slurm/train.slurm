#!/bin/bash
#  salloc -A gts-szonouz6 -N1 --mem-per-gpu=12G -qinferno -t0:15:00 --gres=gpu:V100:1
#SBATCH -JICS-Train
#SBATCH  -Agts-szonouz6
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-gpu=12G

#SBATCH -t2:00:00
#SBATCH -qinferno

source $HOME/ics-venv/bin/activate

cd $HOME/ICS-Attack-Detection/
srun python main.py configs/swat.yaml
