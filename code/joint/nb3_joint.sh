#!/bin/bash

#SBATCH --job-name=norbert3
#SBATCH --account=ec30
#SBATCH --time=04:00:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=7G
#SBATCH --output=/slurm_out/%j.out

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset
export LMOD_DISABLE_SAME_NAME_AUTOSWAP=no

module purge
# module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# module load nlpl-nlptools/01-foss-2024a-Python-3.12.3.lua
# module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3.lua
# module load nlpl-accelerate/1.9.0-foss-2024a-Python-3.12.3.lua
# module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3.lua
# module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3.lua
# module load nlpl-wandb/0.21.4-foss-2024a-Python-3.12.3.lua
source myenv/bin/activate

python3 absa_joint_bert.py --threshold 0.5 --size "large" --lr 5e-05 --batch 8 --train_domain "norpac" --test_domain "norpac" --seed $SEED --hd 0.1 --wd 0.01 --wr 0.0  # change args if needed

