#!/bin/bash

#SBATCH --job-name=norbert3
#SBATCH --account=ec30
#SBATCH --mail-type=FAIL
#SBATCH --time=03:00:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=7G
#SBATCH --output=slurm_out/%j.out

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset
export LMOD_DISABLE_SAME_NAME_AUTOSWAP=no

module purge
#module --force swap StdEnv Zen2Env
#module use -a /fp/projects01/ec30/software/easybuild/modules/all/
#module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
#module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
#module load nlpl-accelerate/0.34.2-foss-2022b-Python-3.10.8
#module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8.lua

source myenv/bin/activate

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
# by default, pass on any remaining command-line options
python3 polarity_bert.py --threshold 0.5 --size "large" --lr 3e-05 --batch 8 --train_domain "norpac_comment_titles" --test_domain "norpac_comment_titles" --gold "False" --seed $SEED --hd 0.1 --wd 0.01 --wr 0.0  # change args if needed

