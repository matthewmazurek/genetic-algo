from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from callbacks import Logger, ReportBestFitness
from checkpoint import CheckPointManager
from custom_types import T_Population
from fn import (fitness_func, fitness_sharing, random_mutation,
                single_point_crossover, tournament_selection)
from ga import GeneticAlgorithm
from population import random_beta_population
