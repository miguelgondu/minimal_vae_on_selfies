"""
Implements a basic evolution strategy using torch,
and tests it on the objective functions.
"""
from typing import Callable, Tuple

import torch
from torch.distributions import MultivariateNormal

from .evolutionary_strategy import EvolutionaryStrategy


class SimpleEvolutionStrategy(EvolutionaryStrategy):
    def __init__(
        self,
        objective_function: Callable[[torch.Tensor], torch.Tensor],
        initial_best: torch.Tensor,
        solution_length : int,
        limits : Tuple[float, float] = None,
        population_size: int = 100,
        exploration: float = 1.0,
    ) -> None:
        super().__init__(objective_function, population_size)

        # Setting up the initial mean and fixed covariance
        self.best = initial_best
        self.covar = exploration * torch.eye(solution_length)

        # Setting up the population's distribution
        self.population_distribution = MultivariateNormal(
            loc=self.best, covariance_matrix=self.covar
        )

    def get_current_best(self) -> torch.Tensor:
        return self.best

    def get_population(self) -> torch.Tensor:
        return self.population_distribution.sample((self.population_size,))

    def step(self) -> torch.Tensor:
        """
        Samples from the current population distribution,
        and saves the best candidate in the mean. It also returns
        the samples.
        """
        # Sample and evaluate
        samples = self.population_distribution.sample((self.population_size,))
        fitnesses = self.objective_function(samples)

        # next one's the best in fitness
        self.best = samples[torch.argmax(fitnesses)]

        # Re-defining the population distribution
        self.population_distribution = MultivariateNormal(
            loc=self.best, covariance_matrix=self.covar
        )

        return samples
