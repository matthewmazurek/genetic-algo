from typing import Any, Protocol

import numpy as np

from .custom_types import T_Population


class PopulationGenerator(Protocol):
    def __call__(self, shape: Any, *args: Any, **kwargs: Any) -> T_Population:
        ...


def random_beta_population(shape: Any, alpha_beta: int | float = 3) -> T_Population:
    """
    Generate a population that is ditributed as a symmetric beta distribution from [-1, 1].

    Parameters:
        shape (Any): The shape of the output array.
        alpha_beta (int | float, optional): The alpha and beta parameters of the beta distribution. Default is 3.

    Returns:
        np.ndarray: An array of random numbers generated from the symmetric beta distribution.

    """
    return np.array(np.random.beta(alpha_beta, alpha_beta, shape)) * 2 - 1
