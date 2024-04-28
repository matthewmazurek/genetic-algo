# Genetic Algorithm Project

This project implements a robust and adaptable Genetic Algorithm (GA) in Python, designed to offer flexibility and extendability for various optimization tasks. By leveraging genetic operations such as selection, crossover, mutation, and fitness evaluation, this project provides a framework that can be tailored to solve complex optimization problems.

## Features

- **Modular Design**: Easily replace or modify components such as fitness functions, selection strategies, and more.
- **Callback System**: Utilize callbacks for logging, monitoring, or modifying the algorithm's execution at runtime.
- **Extensible**: The design allows for easy integration with other Python libraries and custom extensions.

## Files Description

- **`ga.py`**: Main file containing the `GeneticAlgorithm` class which orchestrates the genetic algorithm operations.
- **`fn.py`**: Defines the `Func` class along with its subclasses to encapsulate fitness, selection, mutation, and crossover strategies.
- **`callbacks.py`**: Houses the `CallBack` class and its derivatives, facilitating user-defined operations at various stages of the algorithm.
- **`population.py`**: Contains utility functions for population management tasks within a genetic algorithm context.

## Installation

Clone this repository to your local machine using:
```
git clone https://github.com/matthewmazurek/genetic-algorithm.git
```

## Usage

The `GeneticAlgorithm` class is central to the implementation. You must provide it with necessary configurations including an initial population, fitness function, selection function, mutation function, and crossover function. Optional callbacks can also be provided to enhance the execution flow.

### Basic Example

Below is a basic example demonstrating how to set up and run the genetic algorithm:

```python
from ga import GeneticAlgorithm
from fn import fitness_func, single_point_crossover, random_mutation, tournament_selection
from population import random_beta_population
from callbacks import ReportBestFitness
from functools import partial
import numpy as np

# Define a placeholder fitness function
def dummy_fitness_func(ga, *, ideal_genomes):
    interaction = np.dot(ga.population, ideal_genomes.T)
    sigmoid_fitness = 1 / (1 + np.exp(-interaction))
    return np.max(sigmoid_fitness, axis=1)

# Generate ideal genomes for a simple fitness evaluation
def generate_ideal_genomes(n_genomes, n_ideal_genes, n_genes):
    genomes = np.zeros((n_genomes, n_genes))
    selected_genes = np.random.choice(n_genes, size=(n_genomes, n_ideal_genes), replace=False)
    values = np.random.choice([-1, 1], size=(n_genomes, n_ideal_genes))
    for i in range(n_genomes):
        genomes[i, selected_genes[i]] = values[i]
    return genomes

# Setup GeneticAlgorithm instance with provided functions and configurations
n_genes = 100
n_population = 100
n_generations = 100

ideal_genomes = generate_ideal_genomes(n_genomes=2, n_ideal_genes=3, n_genes=n_genes)
base_fitness_func = partial(dummy_fitness_func, ideal_genomes=ideal_genomes)

ga = GeneticAlgorithm(
    n_generations=n_generations,
    init_population=random_beta_population(shape=(n_population, n_genes)),
    fitness_func=partial(fitness_func, base_fitness_func=base_fitness_func),
    selection_func=tournament_selection,
    crossover_func=partial(single_point_crossover, crossover_rate=0.5),
    mutation_func=partial(random_mutation, population_generating_func=random_beta_population, mutation_rate=0.01)
)

# Execute the algorithm with monitoring
ga.run(callbacks=[ReportBestFitness()])
```

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests to the `main` branch. Alternatively, open issues for bugs, feature requests, or other discussions.

## License

This project is released under the MIT License.