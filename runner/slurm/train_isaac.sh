#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c32
#SBATCH --output=tmp/map-elites-%j.log

srun python -m train_isaac \
              --random_init=256 \
              --random_init_batch=512 \
              --max_evals=100000000 \
              --mutation_op=gaussian_mutation \
              --crossover_op=iso_dd \
              --proportion_evo=1.0 \
              --eval_batch_size=512 \
              --num_gpus=1 \
              --num_evaluators=1 \
              --n_niches=1024 \
              --mutations_per_policy=100 \
              --num_envs_per_policy=1