from typing import Callable, Tuple

import torch
from torch.distributions import Uniform

from utils.wrappers.counters import counted


class RandomSearch:
    """
    A search algorithm that samples a single point uniformly at
    random at each call.
    """

    def __init__(
        self,
        objective_function: Callable[[torch.Tensor], torch.Tensor],
        max_iterations: int,
        limits: Tuple[float, float] = None,
    ) -> None:
        # Wrapping the objective function with the same counter
        # and storing it.
        @counted
        def counted_objective_function(inputs: torch.Tensor):
            return objective_function(inputs)

        self.objective_function = counted_objective_function
        self.max_iterations = max_iterations
        self.limits = limits

        self.trace = None
        self.obj_values = None

    def step(self):
        """
        Samples a point uniformly at random within the specified limits
        """
        # Sample from the [limits]^2 square.
        low, high = self.limits
        next_point = Uniform(
            torch.Tensor([low, low]), torch.Tensor([high, high])
        ).sample()
        obj_value = self.objective_function(next_point)

        # Store the point and the evaluation (optional,
        # we don't need to maintain this necessarily)
        if self.trace is None:
            self.trace = next_point.unsqueeze(0)
        else:
            self.trace = torch.vstack([self.trace, next_point])

        if self.obj_values is None:
            self.obj_values = torch.Tensor([obj_value])
        else:
            self.obj_values = torch.cat([self.obj_values, torch.Tensor([obj_value])])

    def get_trace(self) -> torch.Tensor:
        return self.trace

    def get_current_best(self) -> torch.Tensor:
        return self.trace[torch.argmax(self.obj_values)]
