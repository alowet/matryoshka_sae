#!/bin/bash

#SBATCH --job-name=matryoshka_sae_loop
#SBATCH --output=slurm/out/%J_%x_%a_%t.out
#SBATCH --error=slurm/err/%J_%x_%a_%t.err
#SBATCH --time=0-1
#SBATCH --partition=test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem 8G

# loop over layers 10, 18, and 21
for layer in 10 18 21;
do
    export LAYER=$layer
    if [ $layer -ne 5 ] && [ $layer -ne 8 ]; then
        sbatch --export=ALL,LAYER main.sh
    fi
done
