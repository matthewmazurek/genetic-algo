from unittest.mock import Mock

import numpy as np
import pytest

from fn import (fitness_func, fitness_sharing, random_mutation,
                single_point_crossover, tournament_selection)
from ga import GeneticAlgorithm
from pop_init import PopulationGenerator


def Mock_GA() -> GeneticAlgorithm:
    params = {
        'init_population': np.random.randn(10, 10),
        'fitness_func': Mock(),
        'selection_func': Mock(),
        'mutation_func': Mock(),
        'crossover_func': Mock(),
        'n_generations': 100,
        'keep_parents': False,
        'gene_labels': None,
    }
    return GeneticAlgorithm(**params)

# Test fitness_func


def test_fitness_func():
    ga = Mock_GA()
    def base_fitness_func(ga): return np.ones(ga.population.shape[0])
    l1_weight = 0.1
    fitness_sharing_func = None

    fitness = fitness_func(ga, base_fitness_func=base_fitness_func,  # type: ignore
                           l1_weight=l1_weight, fitness_sharing=fitness_sharing_func)  # type: ignore

    assert isinstance(fitness, np.ndarray)
    assert fitness.shape == ga.population.shape[:1]


def test_fitness_func_with_sharing_and_l1_weight():
    ga = Mock_GA()
    def base_fitness_func(ga): return np.ones(ga.population.shape[0])
    l1_weight = 0.1
    def fitness_sharing_func(ga): return np.ones(ga.population.shape[0])

    fitness = fitness_func(ga, base_fitness_func=base_fitness_func,  # type: ignore
                           l1_weight=l1_weight, fitness_sharing=fitness_sharing_func)  # type: ignore

    assert isinstance(fitness, np.ndarray)
    assert fitness.shape == ga.population.shape[:1]

# Test fitness_sharing


def test_fitness_sharing():
    ga = Mock_GA()

    shared_fitness = fitness_sharing(ga)

    assert isinstance(shared_fitness, np.ndarray)
    assert shared_fitness.shape == (ga.population.shape[0],)

# Test tournament_selection


def test_tournament_selection():
    ga = Mock_GA()

    selected_parents = tournament_selection(ga)

    assert isinstance(selected_parents, list)
    assert len(selected_parents) == 2


def test_tournament_selection_with_custom_parameters():
    ga = Mock_GA()

    selected_parents = tournament_selection(
        ga, num_parents=3, tournament_size=4)

    assert isinstance(selected_parents, list)
    assert len(selected_parents) == 3

# Test single_point_crossover


def test_single_point_crossover():
    ga = Mock_GA()

    child1, child2 = single_point_crossover(ga, genomes=ga.population)

    assert isinstance(child1, np.ndarray)
    assert child1.shape == ga.population.shape[1:]
    assert isinstance(child2, np.ndarray)
    assert child2.shape == ga.population.shape[1:]


# Test random_mutation


def test_random_mutation():
    ga = Mock_GA()
    ga.population = np.random.rand(10, 5)

    def random_generator(shape): return np.array(np.random.randn(*shape))

    mutated_population = random_mutation(
        ga, genomes=ga.population, population_generating_func=random_generator, mutation_rate=0.1)  # type: ignore

    assert isinstance(mutated_population, np.ndarray)
    assert mutated_population.shape == ga.population.shape


def test_random_mutation_with_invalid_mutation_rate():
    ga = Mock_GA()
    ga.population = np.random.rand(10, 5)

    with pytest.raises(ValueError):
        mutated_population = random_mutation(
            ga, genomes=ga.population, population_generating_func=PopulationGenerator, mutation_rate=1.5)  # type: ignore
