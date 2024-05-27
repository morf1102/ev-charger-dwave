import random
from typing import List, Tuple, Optional, Union
from itertools import combinations

import numpy as np
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from scipy.optimize import minimize
from scipy.spatial.distance import cityblock

from .evcp import ECVP


class QuantumAnnealing(ECVP):
    """
    Quantum annealing to solve the problem of finding the optimal locations
    """

    def __init__(
        self,
        width: int,
        height: int,
        num_poi: int,
        num_cs: int,
        num_new_cs: int,
        hyperparams: np.array = np.array([4, 3, 3, 3]),
        sampler: Union[SimulatedAnnealingSampler] = SimulatedAnnealingSampler(),
        num_reads: Optional[int] = None,
        num_sweeps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the quantum annealing algorithm.

        Args:
            width (int): Width of the grid.
            height (int): Height of the grid.
            num_poi (int): Number of points of interest.
            num_cs (int): Number of existing charging stations.
            num_new_cs (int): Number of new charging stations to place.
            num_reads (int, optional): Number of reads. Defaults to 1000.
            num_sweeps (int, optional): Number of sweeps. Defaults to 1000.
            seed (int, optional): Random seed. Defaults to None.
        """
        # Random seed
        random.seed(seed)

        super().__init__(width, height, num_poi, num_cs, num_new_cs)

        self.hyperparams = hyperparams
        self.sampler = sampler
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps

    def build_bqm(self, hyperparams=None):
        """Build bqm that models our problem scenario for the hybrid sampler.

        Args:
            potential_new_cs_nodes (list of tuples of ints):
                Potential new charging locations
            num_poi (int): Number of points of interest
            pois (list of tuples of ints): A fixed set of points of interest
            num_cs (int): Number of existing charging stations
            charging_stations (list of tuples of ints):
                Set of current charging locations
            num_new_cs (int): Number of new charging stations desired

        Returns:
            bqm_np (BinaryQuadraticModel): QUBO model for the input scenario
        """

        a, b, c, d = hyperparams if hyperparams is not None else self.hyperparams

        # Tunable parameters
        alpha = len(self.potential_new_cs_nodes) * a  # 4
        beta = len(self.potential_new_cs_nodes) / b  # 3
        gamma = len(self.potential_new_cs_nodes) * c  # 1.7
        delta = len(self.potential_new_cs_nodes) ** d  # 3

        # Build BQM using adjVectors to find best new charging location s.t. min
        # distance to POIs and max distance to existing charging locations
        bqm = dimod.BinaryQuadraticModel(len(self.potential_new_cs_nodes), "BINARY")

        # Constraint 1: Min average distance to POIs
        if self.num_poi > 0:
            for i, cand_loc in enumerate(self.potential_new_cs_nodes):
                # Compute average distance to POIs from this node
                avg_dist = sum(cityblock(cand_loc, loc) for loc in self.pois)
                bqm.add_linear(i, avg_dist * np.log2(avg_dist) * alpha)

        # Constraint 2: Max distance to existing chargers
        if self.num_cs > 0:
            for i, cand_loc in enumerate(self.potential_new_cs_nodes):
                # Compute average distance to POIs from this node
                avg_dist = sum(
                    cityblock(cand_loc, loc) for loc in self.charging_stations
                )
                bqm.add_linear(i, avg_dist * beta)

        # Constraint 3: Max distance to other new charging locations
        if self.num_new_cs > 1:
            for (i, ai), (j, aj) in combinations(
                enumerate(self.potential_new_cs_nodes), 2
            ):
                bqm.add_interaction(i, j, -cityblock(ai, aj) * gamma)

        # Constraint 4: Choose exactly num_new_cs new charging locations
        bqm.update(
            dimod.generators.combinations(
                bqm.variables, self.num_new_cs, strength=delta
            )
        )
        return bqm

    def run_bqm_and_collect_solutions(self, bqm, **kwargs):
        """Solve the bqm with the provided sampler to find new charger locations.

        Args:
            bqm (BinaryQuadraticModel): The QUBO model for the problem instance
            sampler: Sampler or solver to be used
            potential_new_cs_nodes (list of tuples of ints):
                Potential new charging locations
            **kwargs: Sampler-specific parameters to be used

        Returns:
            new_charging_nodes (list of tuples of ints):
                Locations of new charging stations
        """

        sampleset = self.sampler.sample(bqm, num_reads=self.num_reads, **kwargs)
        ss = sampleset.first.sample
        new_charging_nodes = [
            self.potential_new_cs_nodes[k] for k, v in ss.items() if v == 1
        ]
        return sampleset, new_charging_nodes

    def score_sampleset(self, sampleset) -> float:
        """
        Score the sampleset based on the distance to POIs and existing charging stations.
        """
        n_samples = len(sampleset)

        new_charging_nodes = [
            [self.potential_new_cs_nodes[k] for k, v in sample.items() if v == 1]
            for sample in sampleset
        ]

        score = np.zeros(n_samples)
        for i, new_cs in enumerate(new_charging_nodes):
            score[i] = self.fitness(new_cs)
        total_score = np.mean(score) + np.var(score)
        return total_score

    def search_hyperparams(self, init_guess):
        """
        Search for the best hyperparameters for the quantum annealing algorithm.

        Args:
            init_guess (np.array): Initial guess for the hyperparameters.

        Returns:
            np.array: Best hyperparameters found.
        """
        init_guess += 100

        def objective(hyperparams):
            bqm = self.build_bqm(hyperparams=hyperparams - 100)
            sampleset, _ = self.run_bqm_and_collect_solutions(bqm)
            score = self.score_sampleset(sampleset)
            return -score

        best_result = minimize(
            objective, init_guess, method="Nelder-Mead", bounds=[(101, 1000)] * 4
        )
        best_params = best_result.x - 100
        return best_params


class GeneticAlgortihm(ECVP):
    """
    Genetic algorithm to solve the problem of finding the optimal locations
    for new charging stations.
    """

    def __init__(
        self,
        width: int,
        height: int,
        num_poi: int,
        num_cs: int,
        num_new_cs: int,
        pop_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        generations: int = 500,
    ) -> None:
        """
        Initialize the genetic algorithm.

        Args:
            width (int): Width of the grid.
            height (int): Height of the grid.
            num_poi (int): Number of points of interest.
            num_cs (int): Number of existing charging stations.
            num_new_cs (int): Number of new charging stations to place.
            pop_size (int, optional): Population size. Defaults to 100.
            mutation_rate (float, optional): Mutation rate. Defaults to 0.1.
            crossover_rate (float, optional): Crossover rate. Defaults to 0.5.
            generations (int, optional): Number of generations. Defaults to 500.
            seed (int, optional): Random seed. Defaults to None.
        """
        super().__init__(width, height, num_poi, num_cs, num_new_cs)

        # Set up parameters
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population: List[List[Tuple[int, int]]] = []
        self.best_fitness = 0
        self.best_generation = 0

    def init_population(self) -> None:
        """
        Initialize the population with random coordinates.
        """
        self.population = [
            random.choices(self.potential_new_cs_nodes, k=self.num_new_cs)
            for _ in range(self.pop_size)
        ]

    def tournament_selection(self, k: int = 2) -> List[Tuple[int, int]]:
        """
        Perform a tournament selection to select a parent.

        Args:
            k (int, optional): Number of individuals to select. Defaults to 2.

        Returns:
            List[Tuple[int, int]]: The selected parent.
        """
        selected = random.choices(self.population, k=k)
        return max(selected, key=self.fitness)

    def crossover(
        self, parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Perform a crossover operation between two parents.

        Args:
            parent1 (List[Tuple[int, int]]): The first parent.
            parent2 (List[Tuple[int, int]]): The second parent.

        Returns:
            List[Tuple[int, int]]: The crossover child.
        """
        child: List[Tuple[int, int]] = []
        for i in range(self.num_new_cs):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])

        # Ensure uniqueness of genes (nodes) in the child
        unique_child = [tuple(node) for node in np.unique(child, axis=0).tolist()]

        # Fill in the remaining genes from either parent to maintain the correct length
        parent_combined = parent1 + parent2
        for node in parent_combined:
            if len(unique_child) < self.num_new_cs and node not in unique_child:
                unique_child.append(node)

        return unique_child

    def mutate(self, child: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Mutate a child by randomly changing the coordinates, ensuring they are not on POIs.

        Args:
            child (List[Tuple[int, int]]): The child to mutate.

        Returns:
            List[Tuple[int, int]]: The mutated child.
        """
        if random.random() < self.mutation_rate:
            mutated_child: List[Tuple[int, int]] = []
            for node in child:
                new_node = node
                while True:
                    # Mutate x and y coordinates separately
                    new_x = node[0] + random.randint(-1, 1)
                    new_y = node[1] + random.randint(-1, 1)
                    # Ensure new coordinates are within grid bounds
                    new_x = max(0, min(new_x, self.width - 1))
                    new_y = max(0, min(new_y, self.height - 1))
                    new_node = (new_x, new_y)
                    # Check if the new node is a POI
                    if (
                        new_node
                        not in self.pois + self.charging_stations + mutated_child
                    ):
                        break
                mutated_child.append(new_node)
            return mutated_child
        return child

    def run(self) -> None:
        """
        Run the genetic algorithm to solve the problem.
        """
        # Initialize the population
        self.init_population()
        for gen in range(self.generations):
            new_population: List[List[Tuple[int, int]]] = []
            for _ in range(self.pop_size):
                parent1 = self.tournament_selection()
                if random.random() < self.crossover_rate:
                    parent2 = self.tournament_selection()
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
            for candidate in self.population:
                fit = self.fitness(candidate)
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.new_charging_nodes = candidate
                    self.best_generation = gen
        self.new_charging_nodes = [tuple(node) for node in self.new_charging_nodes]
