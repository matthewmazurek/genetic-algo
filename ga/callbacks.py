import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    from .ga import GeneticAlgorithm


class CallBack:

    def on_run_start(self, ga: 'GeneticAlgorithm') -> Any:
        pass

    def on_run_end(self, ga: 'GeneticAlgorithm') -> Any:
        pass

    def on_iter_start(self, ga: 'GeneticAlgorithm') -> Any:
        pass

    def on_iter_end(self, ga: 'GeneticAlgorithm') -> Any:
        pass


@dataclass
class Logger(CallBack):

    """
    A class that provides logging functionality for a GeneticAlgorithm.

    This class keeps track of various information during the execution of a GeneticAlgorithm,
    such as generation number, population, fitness values, and timestamps.

    Attributes:
        log (list): A list of dictionaries representing the logged information.
        run_parameters (dict): A dictionary containing the parameters used for the GeneticAlgorithm run.

    Methods:
        on_run_start(ga: GeneticAlgorithm) -> None:
            Called when the GeneticAlgorithm run starts.
            Initializes the log and sets the start time.

        on_run_end(ga: GeneticAlgorithm) -> None:
            Called when the GeneticAlgorithm run ends.
            Sets the stop time.

        on_iter_start(ga: GeneticAlgorithm) -> None:
            Called at the start of each iteration.
            Appends a new log item with generation number, population, and start time.

        on_iter_end(ga: GeneticAlgorithm) -> None:
            Called at the end of each iteration.
            Updates the last log item with fitness values and stop time.

        append(dict) -> None:
            Appends additional information to the last log item.

        get_log(generation=None) -> list:
            Returns the log items for a specific generation.
            If generation is not specified, returns the entire log.

        get_run_parameters(ga: GeneticAlgorithm) -> dict:
            Returns a dictionary of run parameters used for the GeneticAlgorithm.

        serialize_object(obj) -> str:
            Converts various types of objects into a more human-readable string format.

        get_runtime() -> datetime.timedelta:
            Returns the runtime of the GeneticAlgorithm run.

    """

    def on_run_start(self, ga: 'GeneticAlgorithm') -> None:

        self.run_parameters = Logger.get_run_parameters(ga)

        if not hasattr(self, 'log'):
            self.log: List[Dict[str, Any]] = []

        if not hasattr(self.run_parameters, 'start_time'):
            self.run_parameters['start_time'] = datetime.datetime.now()

    def on_run_end(self, ga: 'GeneticAlgorithm') -> None:
        self.run_parameters['stop_time'] = datetime.datetime.now()

    def on_iter_start(self, ga: 'GeneticAlgorithm') -> None:
        self.log.append({
            'generation': ga.generation,
            'population': ga.population,
            'start_time': datetime.datetime.now(),
        })

    def on_iter_end(self, ga: 'GeneticAlgorithm') -> None:
        self.log[-1].update({
            'fitnesses': ga.fitnesses,
            'stop_time': datetime.datetime.now(),
        })

    def append(self, dict):
        self.log[-1].update(dict)

    def get_log(self, generation=None):
        if generation is not None:
            return [log_item for log_item in self.log if log_item['generation'] == generation]
        else:
            return self.log

    def get_runtime(self):
        if 'start_time' not in self.run_parameters:
            raise ValueError("Run has not started yet.")

        if 'stop_time' not in self.run_parameters:
            raise ValueError("Run has not ended yet.")

        assert isinstance(self.run_parameters['start_time'], datetime.datetime)
        assert isinstance(self.run_parameters['stop_time'], datetime.datetime)

        return self.run_parameters['stop_time'] - self.run_parameters['start_time']

    @staticmethod
    def get_run_parameters(ga: 'GeneticAlgorithm'):
        props = ['n_generations', 'fitness_func', 'selection_func',
                 'keep_parents', 'mutation_func', 'crossover_func', 'gene_labels']
        return {prop: Logger.serialize_object(getattr(ga, prop)) for prop in props}

    @staticmethod
    def serialize_object(obj):
        """ Converts various types of objects into a more human-readable string format. """
        if callable(obj):
            # Returns the function's name and details if possible
            return repr(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to list for easier readability
        elif hasattr(obj, '__dict__'):
            # Recursively serialize objects
            return {key: Logger.serialize_object(value) for key, value in obj.__dict__.items()}
        else:
            return str(obj)


@dataclass
class ReportBestFitness(CallBack):
    frequency: int = 10

    def on_iter_end(self, ga: 'GeneticAlgorithm') -> None:

        if ga.generation % self.frequency != 0 and ga.generation != ga.n_generations:
            return

        best_fitness = ga.best()['fitness']
        print(
            f"Generation {ga.generation:03d} | Best fitness: {best_fitness:2f}")


@dataclass
class EarlyStopping(CallBack):
    patience: int = 3
    record: float = 0
    counter: int = 0

    def on_iter_end(self, ga: 'GeneticAlgorithm') -> str | None:

        best_fitness = ga.best()['fitness']
        if self.record < best_fitness:
            self.record = best_fitness
            self.counter = 0
        else:
            self.counter += 1

        if self.counter > self.patience:
            print(
                f"Early stopping at generation {ga.generation:03d} | Best fitness: {best_fitness:2f}")
            return 'stop'  # early stopping
