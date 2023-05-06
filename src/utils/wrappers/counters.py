from typing import Callable

import torch


def counted(obj_function: Callable[[torch.Tensor], torch.Tensor]):
    """
    Counts on how many points the obj function was evaluated
    """

    def wrapped(inputs):
        wrapped.calls += 1

        if len(inputs.shape) == 1:
            # We are evaluating at a single point [x, y]
            wrapped.n_points += 1
        else:
            # We are evaluating at a several points.
            wrapped.n_points += len(inputs)

        return obj_function(inputs)

    wrapped.calls = 0
    wrapped.n_points = 0
    return wrapped
