"""
A common interface for evolutionary strategies.
"""
from typing import Callable

import torch

from utils.wrappers.counters import counted


class EvolutionaryStrategy:
    def __init__(
        self,
        objective_function: Callable[[torch.Tensor], torch.Tensor],
        population_size: int,
    ):
        # Wraps the objective function in a counter. After this
        # we can access self.objective_function.calls and
        # self.objective_function.n_points to check how
        # many calls and in how many points the function
        # was evaluated
        @counted
        def counted_objective_function(inputs: torch.Tensor):
            return objective_function(inputs)

        self.objective_function = counted_objective_function
        self.population_size = population_size

    def step(self):
        """
        Takes a step of the evolutionary strategy.
        """
        raise NotImplementedError

    def get_current_best(self):
        """
        Returns the current elite of the population.
        """
        raise NotImplementedError

    def get_population(self):
        """
        Returns the current population.
        """
        raise NotImplementedError
