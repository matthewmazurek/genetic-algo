import datetime
import os
import pickle
from dataclasses import dataclass

from .callbacks import CallBack, Logger
from .ga import GeneticAlgorithm


@dataclass
class CheckPointManager(CallBack):
    """
    A class that manages checkpoints for a GeneticAlgorithm.

    Attributes:
        ga (GeneticAlgorithm): The GeneticAlgorithm object associated with the checkpoint manager.
        logger (Logger): The logger object used for logging information during the GeneticAlgorithm run.
        name (str, optional): The name of the checkpoint. If not provided, a default name will be generated based on the current date and time.
        directory (str): The directory where the checkpoints will be saved.
        save_frequency (int): The frequency at which checkpoints will be saved during the GeneticAlgorithm run.

    Methods:
        save(file_name=None): Saves the current state of the logger and associated information to a checkpoint file.
        load(file_name): Loads the state of the logger and associated information from a checkpoint file.
        _update_ga(): Updates the GeneticAlgorithm object with the latest information from the logger.
        _update_logger(data): Updates the logger object with the provided data.
        assert_ga_consistency(ga): Asserts that the GeneticAlgorithm object passed to the callback is consistent with the one associated with the checkpoint manager.
        on_iter_end(ga): Callback function called at the end of each iteration of the GeneticAlgorithm run.
        on_run_end(ga): Callback function called at the end of the GeneticAlgorithm run.
    """

    ga: GeneticAlgorithm
    logger: Logger
    name: str | None = None
    directory: str = '.'
    save_frequency: int = 0

    def __post_init__(self):
        if self.name is None:
            current_datetime = datetime.datetime.now()
            self.name = f'{self.ga.__class__.__name__}_{current_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'

    def save(self, file_name: str | None = None):
        """
        Saves the current state of the logger and associated information to a checkpoint file.

        Args:
            file_name (str, optional): The name of the checkpoint file. If not provided, a default name will be generated based on the checkpoint manager's name and the current generation of the GeneticAlgorithm.

        Returns:
            None
        """
        os.makedirs(self.directory, exist_ok=True)

        if file_name is None:
            file_name = os.path.join(
                self.directory, f'{self.name}_gen_{self.ga.generation:03d}.pkl')

        with open(file_name, 'wb') as file:
            data = vars(self.logger)
            data.update({
                'checkpoint_name': self.name,
            })
            pickle.dump(data, file)

    def load(self, file_name):
        """
        Loads the state of the logger and updates the associated GeneticAlgorithm object.

        Args:
            file_name (str): The name of the checkpoint file to load.

        Returns:
            None
        """
        with open(file_name, 'rb') as file:
            data = pickle.load(file)

        if 'checkpoint_name' in data:
            self.name = data['checkpoint_name']
            del data['checkpoint_name']

        self._update_logger(data)
        self._update_ga()

    def _update_ga(self):
        """
        Updates the GeneticAlgorithm object with the latest information from the logger.

        Returns:
            None
        """
        latest = self.logger.log[-1]
        for key, value in latest.items():
            if key in ['generation', 'population', 'fitnesses']:
                setattr(self.ga, key, value)

    def _update_logger(self, data):
        """
        Updates the logger object with the provided data.

        Args:
            data (dict): The data to update the logger with.

        Returns:
            None
        """
        self.logger.__dict__ = data

    def assert_ga_consistency(self, ga: GeneticAlgorithm):
        """
        Asserts that the GeneticAlgorithm object passed to the callback is consistent with the one associated with the checkpoint manager.

        Args:
            ga (GeneticAlgorithm): The GeneticAlgorithm object to check consistency with.

        Raises:
            ValueError: If the GeneticAlgorithm object is not consistent with the one associated with the checkpoint manager.

        Returns:
            None
        """
        if self.ga != ga:
            raise ValueError(
                'CheckPointManager was instantiated on a different GeneticAlgorithm object than the one passed to the callback.')

    def on_iter_end(self, ga: 'GeneticAlgorithm') -> None:
        """
        Save a checkpoint at the end of every (save_frequency)th generation of the GeneticAlgorithm run.

        Args:
            ga (GeneticAlgorithm): The GeneticAlgorithm object associated with the callback.

        Returns:
            None
        """
        self.assert_ga_consistency(ga)

        if self.save_frequency > 0 and ga.generation % self.save_frequency == 0:
            self.save()

    def on_run_end(self, ga: 'GeneticAlgorithm') -> None:
        """
        Save a checkpoint at the end of the GeneticAlgorithm run.

        Args:
            ga (GeneticAlgorithm): The GeneticAlgorithm object associated with the callback.

        Returns:
            None
        """
        self.assert_ga_consistency(ga)
        self.save()
