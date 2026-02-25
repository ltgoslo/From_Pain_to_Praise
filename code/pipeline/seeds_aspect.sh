#!/bin/bash

for SEED in 42 43 44 45 46
do
    echo "Submitting aspect job with seed $SEED"
    sbatch --export=ALL,SEED=$SEED nb3_run_a100.sh
done
