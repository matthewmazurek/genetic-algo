from typing import TYPE_CHECKING, Any, List, Protocol, Tuple

import numpy as np

from .custom_types import T_Genome, T_Population

if TYPE_CHECKING:
    from .ga import GeneticAlgorithm
    from .population import PopulationGenerator


class Func(Protocol):
    def __call__(self, first: 'GeneticAlgorithm', *args: Any, **Kwargs: Any) -> Any:
        ...


def fitness_func(ga: 'GeneticAlgorithm', *,
                 base_fitness_func: Func,
                 l1_weight: float | None = 0.1,
                 fitness_sharing: Func) -> np.ndarray:
    """
    Calculates the fitness of each individual in the genetic algorithm population.

    Args:
        ga (ga.'GeneticAlgorithm'): The genetic algorithm instance.
        base_fitness_func (Func): The base fitness function used to calculate the base fitness of each individual.
        l1_weight (float | None, optional): The weight for the L1 regularization penalty. Defaults to 0.1.
        fitness_sharing (Func, optional): The fitness sharing function used to calculate the shared fitness of each individual.

    Returns:
        np.ndarray: An array containing the fitness values for each individual in the population.
    """
    # Base fitness
    base_fitness = base_fitness_func(ga)

    # Calculate the regularization penalty as the sum of absolute perturbations normalized to genome length
    regularization_penalty = np.sum(
        np.abs(ga.population), axis=1) / ga.population.shape[1] * l1_weight

    # Calculate shared fitness
    if fitness_sharing:
        shared_fitness = fitness_sharing(ga)
    else:
        shared_fitness = 1

    # Calculate the final fitness
    fitness = base_fitness * shared_fitness - regularization_penalty

    return fitness


def fitness_sharing(ga, *, gene_threshold: float = 0.25, sharing_radius: float = 0.5, sharing_level: int | float = 2) -> np.ndarray:
    """
    Calculates the shared fitness coefficient for a genetic algorithm population using fitness sharing.

    Parameters:
    - ga (ga.'GeneticAlgorithm'): The genetic algorithm instance.
    - gene_threshold (float, optional): The threshold value for genes. Genes with absolute values less than this threshold will be set to 0. Default is 0.25.
    - sharing_radius (float, optional): The radius within which genomes are considered to be in the same niche. Default is 0.5.
    - sharing_level (int | float, optional): The level of sharing. Determines the shape of the sharing function. Default is 2.

    Returns:
    - shared_fitness: An array of shared fitness coefficients for each genome in the population.
    """

    # Threshold genomes
    thresholded = ga.population.copy()
    thresholded[np.abs(thresholded) < gene_threshold] = 0

    # Calculate distance between genomes
    pairwise_difference = thresholded[:,
                                      np.newaxis, :] - thresholded[np.newaxis, :, :]
    pairwise_distance = np.linalg.norm(pairwise_difference, axis=2)

    # Normalization factor
    # The normalization factor is the maximum distance between two thresholded genomes
    normalization_factor = np.sqrt(
        np.sum(np.where(pairwise_difference > 0, 2, 0), axis=2))
    normalization_factor = np.maximum(
        normalization_factor, 1)  # Avoid division by zero

    # Calculate normalized pairwise distance
    normalized_pairwise_distance = pairwise_distance / normalization_factor

    # Calculate sharing function
    sharing_function = 1 - (normalized_pairwise_distance /
                            sharing_radius) ** sharing_level
    sharing_function[normalized_pairwise_distance >= sharing_radius] = 0

    # Calculate the shared fitness coefficient
    shared_fitness = 1 / np.sum(sharing_function, axis=1)

    return shared_fitness


def tournament_selection(ga: 'GeneticAlgorithm', *, num_parents: int = 2, tournament_size: int = 3) -> List[int]:
    """
    Selects a specified number of parents from the population using tournament selection.

    Args:
        ga (ga.'GeneticAlgorithm'): The genetic algorithm instance.
        num_parents (int, optional): The number of parents to select. Defaults to 2.
        tournament_size (int, optional): The number of individuals to participate in each tournament. Defaults to 3.

    Returns:
        List[int]: A list of indices representing the selected parents from the population.
    """
    winners = []

    for _ in range(num_parents):
        participants = np.random.choice(
            np.arange(len(ga.population)), size=tournament_size, replace=False)
        participant_fitnesses = ga.fitnesses[participants]
        winner_index = participants[np.argmax(participant_fitnesses)]
        winners.append(winner_index)

    return winners


def single_point_crossover(ga: 'GeneticAlgorithm', *, genomes: T_Population, crossover_rate: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform single-point crossover on the genetic algorithm population.

    Args:
        ga (ga.'GeneticAlgorithm'): The genetic algorithm instance.
        crossover_rate (float, optional): The probability of performing crossover. Defaults to 0.7.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two child arrays resulting from the crossover operation.
    """

    # Ensure there are at least two parents
    n_genomes, n_genes = genomes.shape
    if n_genomes < 2:
        raise ValueError("There must be at least two parents.")

    # Randomly select two distinct parents from the list
    parents = np.random.choice(n_genomes, size=2, replace=False)

    # Perform crossover
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, n_genes)  # Point of crossover
        parent1, parent2 = genomes[parents]
        child1 = np.concatenate(
            [parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate(
            [parent2[:crossover_point], parent1[crossover_point:]])
    else:
        child1, child2 = ga.population[parents]

    return child1, child2


def random_mutation(ga: 'GeneticAlgorithm', *, genomes: T_Population, population_generating_func: 'PopulationGenerator', mutation_rate: float = 0.1) -> np.ndarray:
    """
    Applies random mutation to the individuals in the population of a ga.'GeneticAlgorithm'.

    Args:
        ga (ga.'GeneticAlgorithm'): The ga.'GeneticAlgorithm' instance.
        population_generating_func (PopulationGenerator): A function that generates a new population.
        mutation_rate (float, optional): The probability of an individual being mutated. Must be between 0 and 1. Defaults to 0.1.

    Returns:
        np.ndarray: The mutated population.

    Raises:
        ValueError: If mutation_rate is not within the valid range of 0 to 1.
    """
    # Ensuring mutation_rate is within valid range
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1.")

    mutations = population_generating_func(genomes.shape)
    mutated = genomes.copy()
    rand_idx = np.random.uniform(size=genomes.shape) < mutation_rate
    mutated[rand_idx] = mutations[rand_idx]

    return mutated
