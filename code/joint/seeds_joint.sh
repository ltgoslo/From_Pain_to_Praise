for SEED in 42 43 44 45 46
do
    echo "Submitting aspect job with seed $SEED"
    sbatch --export=ALL,SEED=$SEED nb3_joint.sh
done
