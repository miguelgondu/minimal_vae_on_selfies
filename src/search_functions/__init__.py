"""
This module contains all the search algorithms we plan to use
in our comparison. They were first developed in the REAL Lab [1].

They're exported at this level too, making imports easier.

[1]: Benchmarking evolutionary strategies and Bayesian Optimization.
     Elías Najarro & Miguel González-Duque, 2023.
     https://github.com/real-itu/benchmarking_evolution_and_bo
"""
# Baselines
from .baselines.random_search import RandomSearch

# B.O.
from .bayesian_optimization.bayesian_optimization import BayesianOptimization

# Evolutionary strategies
from .evolutionary_strategies.simple_evolution_strategy import SimpleEvolutionStrategy
from .evolutionary_strategies.snes import SNES
from .evolutionary_strategies.pgpe import PGPE
from .evolutionary_strategies.cma_es import CMA_ES
