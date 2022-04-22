#!/usr/bin/env bash
#SBATCH --gres=gpu:4
#SBATCH -N1
#SBATCH -n1
#SBATCH -c32
#SBATCH --output=tmp/map-elites-%j.log

srun python -m train_isaac \
              --random_init=200 \
              --random_init_batch=500 \
              --max_evals=100000 \
              --mutation_op=gaussian_mutation \
              --crossover_op=iso_dd \
              --proportion_evo=1.0 \
              --eval_batch_size=500 \
              --num_gpus=4 \
              --num_evaluators=4 \
              --n_niches=2048 \
              --num_agents=500