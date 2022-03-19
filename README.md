# Python3 Map-Elites
High throughput version of CVT Map Elites with support for neural network controllers. Currently only works for Bipedal Walker, but other environments will be supported 

Reference implementation of CVT Map Elites: https://github.com/resibots/pymap_elites

Reference implementation of Policy Gradient Assisted Map Elites: https://github.com/ollenilsson19/PGA-MAP-Elites

## About CVT Map Elites

CVT-MAP-Elites can be used instead of the standard MAP-Elites described in:
Mouret JB, Clune J. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.

The general philosophy is to provide implementations (1 page of code) that are easy to transform and include in your own research. This means that there is some code redundancy between the algorithms. If you are interested in a more advanced framework:
- Sferes (C++): https://github.com/sferes2/sferes2 
- QDPy (Python) https://pypi.org/project/qdpy/

By default, the evaluations are parallelized on each core (using the multiprocessing package).



## References:
If you use this code in a scientific paper, please cite:

**Main paper**: Mouret JB, Clune J. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.

**CVT Map-Elites**: Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multi-dimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3.

**Variation operator**: Vassiliades V, Mouret JB. Discovering the Elite Hypervolume by Leveraging Interspecies Correlation. Proc. of GECCO. 2018.

**Multitask-MAP-Elites**: Mouret JB, Maguire G. Quality Diversity for Multi-task Optimization. Proc of GECCO. 2020.

**PGA-MAP-Elites**: Nilsson, Olle, and Antoine Cully. "Policy gradient assisted map-elites." Proceedings of the Genetic and Evolutionary Computation Conference. 2021.

## Installation
```
conda env create -f map-elites.yaml
conda activate map-elites
```

## Basic usage

An example script. From the root directory run 
```python
python -m train_bipedal_walker --num_workers=YOUR_NUM_CPUS --num_gpus=YOUR_NUM_GPUS --mutation_op=gaussian_mutation --crossover_op=iso_dd
```

A full list of parameters can be found in ```train_bipedal_walker``` or by typing ```python -m train_bipedal_walker --help```

