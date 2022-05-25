#!/usr/bin/env bash
#SBATCH --gres=gpu:4
#SBATCH -N1
#SBATCH -n1
#SBATCH -c32
#SBATCH --output=tmp/map-elites-%j.log

srun python -m train_isaac \
              --random_init=64 \
              --random_init_batch=512 \
              --max_evals=1000000 \
              --mutation_op=gaussian_mutation \
              --crossover_op=iso_dd \
              --proportion_evo=1.0 \
              --eval_batch_size=512 \
              --num_gpus=1 \
              --num_evaluators=1 \
              --n_niches=1024 \
              --num_agents=512 \
              --mutations_per_policy=4 \
              --num_envs_per_agent=10