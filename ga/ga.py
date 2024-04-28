
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional,
                    Protocol, Tuple, TypeAlias)

import numpy as np

from .callbacks import CallBack
from .custom_types import T_Population

if TYPE_CHECKING:
    from .fn import Func


@dataclass
class GeneticAlgorithm:

    # Parameters
    init_population: T_Population
    fitness_func: 'Func'
    selection_func: 'Func'
    mutation_func: 'Func'
    crossover_func: 'Func'
    n_generations: int = 100
    keep_parents: bool = False
    gene_labels: List[str] | None = None

    def __post_init__(self):
        self.population = self.init_population
        self.generation = 0
        self.fitnesses = - np.ones_like(self.population) * np.inf

    def run_callbacks(self, hook: str, callbacks: List[CallBack]) -> List[Any]:
        return [getattr(cb, hook)(self) for cb in callbacks]

    def run(self, callbacks: list[CallBack]):

        # Run start
        self.run_callbacks('on_run_start', callbacks)

        while self.generation < self.n_generations:

            # Iter start
            self.run_callbacks('on_iter_start', callbacks)

            # Evaluate fitness
            self.fitnesses = self.fitness_func(self)

            # Selection, Crossover, and Mutation to create the next generation
            new_population = []
            while len(new_population) < len(self.population):

                parent_idxs = self.selection_func(self)
                parents = self.population[parent_idxs]

                if self.keep_parents:
                    new_population.extend(parents)

                parents = self.mutation_func(self, genomes=parents)
                children = self.crossover_func(self, genomes=parents)

                new_population.extend(children)

            # Ensure the population size remains the same
            new_population = new_population[:len(self.population)]

            # Log iteration end
            cbs = self.run_callbacks('on_iter_end', callbacks)

            # Update population and generation
            self.population = np.array(new_population)
            self.generation += 1

            # Early stopping (break if any callback return value is 'stop')
            if 'stop' in cbs:
                break

        self.run_callbacks('on_run_end', callbacks)

    def best(self) -> Dict[str, Any]:

        return {
            'genome': self.population[np.argmax(self.fitnesses)],
            'fitness': self.fitnesses[np.argmax(self.fitnesses)]
        }
