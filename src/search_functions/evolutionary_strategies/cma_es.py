"""
Implements CMA-ES using the common interface we have
for evolutionary strategies.
"""
from typing import Callable, Tuple

import torch

from evotorch import Problem
from evotorch.algorithms import CMAES as CMAES_from_evotorch

from .evolutionary_strategy import EvolutionaryStrategy


class CMA_ES(EvolutionaryStrategy):
    """
    TODO: add docs
    """

    def __init__(
        self,
        objective_function: Callable[[torch.Tensor], torch.Tensor],
        population_size: int,
        exploration: float,
        solution_length: int,
        limits: Tuple[float, float] = None,
    ):
        """
        TODO: add docs
        """
        # Stores it in the class attributes
        super().__init__(objective_function, population_size)

        # Defines the Evotorch problem and searcher.
        self.problem = Problem(
            "max",
            objective_func=self.objective_function,
            initial_bounds=limits,
            solution_length=solution_length,
            vectorized=True,
            dtype=torch.get_default_dtype(),
        )

        self._cmaes_searcher = CMAES_from_evotorch(
            self.problem, stdev_init=exploration, popsize=population_size
        )

    def get_current_best(self):
        return self._cmaes_searcher.get_status_value("pop_best").access_values()

    def get_population(self):
        return self._cmaes_searcher.population.access_values()

    def step(self):
        self._cmaes_searcher.step()
