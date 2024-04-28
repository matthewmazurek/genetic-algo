import datetime
from unittest.mock import Mock

import numpy as np
import pytest

from callbacks import EarlyStopping, Logger, ReportBestFitness
from ga import GeneticAlgorithm


def approx_equal_dt(a: datetime.datetime, b: datetime.datetime) -> bool:
    return abs(a - b) < datetime.timedelta(seconds=1)


def test_ReportBestFitness_on_iter_end():
    ga = Mock()
    ga.generation = 10
    ga.n_generations = 100
    ga.best.return_value = {'fitness': 0.9}

    report_best_fitness = ReportBestFitness()
    report_best_fitness.on_iter_end(ga)

    ga.best.assert_called_once()


def test_ReportBestFitness_on_iter_end_with_frequency():
    ga = Mock()
    ga.generation = 5
    ga.n_generations = 100
    ga.best.return_value = {'fitness': 0.9}

    report_best_fitness = ReportBestFitness(frequency=5)
    report_best_fitness.on_iter_end(ga)

    ga.best.assert_called_once()


def test_EarlyStopping_on_iter_end():
    ga = Mock()
    ga.best.return_value = {'fitness': 0.9}

    early_stopping = EarlyStopping(patience=3, record=0.8, counter=0)
    result = early_stopping.on_iter_end(ga)

    assert result is None
    ga.best.assert_called_once()
    assert early_stopping.record == 0.9
    assert early_stopping.counter == 0


def test_EarlyStopping_on_iter_end_with_counter():
    ga = Mock()
    ga.best.return_value = {'fitness': 0.8}

    early_stopping = EarlyStopping(patience=3, record=0.9, counter=0)
    result = early_stopping.on_iter_end(ga)

    assert result is None
    ga.best.assert_called_once()
    assert early_stopping.record == 0.9
    assert early_stopping.counter == 1


def test_EarlyStopping_on_iter_end_with_counter_and_patience():
    ga = Mock()
    ga.generation = 10
    ga.best.return_value = {'fitness': 0.8}

    early_stopping = EarlyStopping(patience=3, record=0.9, counter=3)
    result = early_stopping.on_iter_end(ga)

    assert result is 'stop'


def test_Logger_on_run_start():
    ga = Mock()
    logger = Logger()

    logger.on_run_start(ga)

    assert hasattr(logger, 'log')
    assert hasattr(logger, 'run_parameters')
    assert 'start_time' in logger.run_parameters and isinstance(
        logger.run_parameters['start_time'], datetime.datetime)
    assert approx_equal_dt(
        logger.run_parameters['start_time'], datetime.datetime.now())


def test_Logger_on_run_end():
    ga = Mock()
    logger = Logger()
    logger.run_parameters = {
        'start_time': datetime.datetime.now() - datetime.timedelta(seconds=10)}

    logger.on_run_end(ga)

    assert 'stop_time' in logger.run_parameters and isinstance(
        logger.run_parameters['stop_time'], datetime.datetime)
    assert approx_equal_dt(
        logger.run_parameters['stop_time'], datetime.datetime.now())


def test_Logger_on_iter_start():
    ga = Mock()
    ga.generation = 10
    ga.population = np.random.uniform(-1, 1, size=(10, 10))
    logger = Logger()
    logger.log = []

    logger.on_iter_start(ga)

    assert len(logger.log) == 1
    assert logger.log[0]['generation'] == ga.generation
    assert 'population' in logger.log[0]
    assert logger.log[0]['population'] is ga.population


def test_Logger_on_iter_end():
    ga = Mock()
    ga.fitnesses = np.random.uniform(-1, 1, size=(10,))
    logger = Logger()
    logger.log = [{
        'generation': 10,
        'population': np.random.uniform(-1, 1, size=(10, 10)),
        'start_time': datetime.datetime.now() - datetime.timedelta(seconds=10)
    }]

    logger.on_iter_end(ga)

    assert 'stop_time' in logger.log[0]
    assert logger.log[0]['stop_time'] > logger.log[0]['start_time']
    assert 'fitnesses' in logger.log[0]
    assert logger.log[0]['fitnesses'] is ga.fitnesses


def test_Logger_append():
    ga = Mock()
    logger = Logger()
    logger.log = [{'generation': 10}]

    logger.append({'additional_info': 'some info'})

    assert 'additional_info' in logger.log[0]
    assert logger.log[0]['additional_info'] == 'some info'


def test_Logger_get_log_with_generation():
    ga = Mock()
    logger = Logger()
    logger.log = [{'generation': 10}, {'generation': 11}, {'generation': 10}]

    log = logger.get_log(generation=10)

    assert len(log) == 2
    assert log[0]['generation'] == 10
    assert log[1]['generation'] == 10


def test_Logger_get_log_without_generation():
    ga = Mock()
    logger = Logger()
    logger.log = [{'generation': 10}, {'generation': 11}, {'generation': 10}]

    log = logger.get_log()

    assert len(log) == 3
    assert log[0]['generation'] == 10
    assert log[1]['generation'] == 11
    assert log[2]['generation'] == 10


def test_Logger_get_run_parameters():
    ga = Mock()
    ga.n_generations = 100
    ga.fitness_func = Mock()
    ga.selection_func = Mock()
    ga.keep_parents = True
    ga.mutation_func = Mock()
    ga.crossover_func = Mock()
    ga.gene_labels = ['A', 'B', 'C']

    logger = Logger()
    run_parameters = logger.get_run_parameters(ga)

    assert 'n_generations' in run_parameters
    assert run_parameters['n_generations'] == repr(ga.n_generations)
    assert 'fitness_func' in run_parameters
    assert run_parameters['fitness_func'] == repr(ga.fitness_func)
    assert 'selection_func' in run_parameters
    assert run_parameters['selection_func'] == repr(ga.selection_func)
    assert 'keep_parents' in run_parameters
    assert run_parameters['keep_parents'] == repr(ga.keep_parents)
    assert 'mutation_func' in run_parameters
    assert run_parameters['mutation_func'] == repr(ga.mutation_func)
    assert 'crossover_func' in run_parameters
    assert run_parameters['crossover_func'] == repr(ga.crossover_func)
    assert 'gene_labels' in run_parameters
    assert run_parameters['gene_labels'] == repr(ga.gene_labels)
