import random
from typing import List, Tuple, Optional, Union
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler
from scipy.optimize import minimize

from .utils import influence_matrix
from .evcp import EVCP


class QuantumAnnealing(EVCP):
    """
    Quantum annealing to solve the problem of finding the optimal locations
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        num_poi: int,
        num_cs: int,
        num_new_cs: int,
        hyperparams: ArrayLike = [2, 6, 7, 2],
        sampler: Union[
            SimulatedAnnealingSampler, LeapHybridSampler
        ] = SimulatedAnnealingSampler(),
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
        # Initialize the scenario
        super().__init__(shape, num_poi, num_cs, num_new_cs, seed=seed)

        # Set up parameters
        self.hyperparams = np.asarray(hyperparams, dtype=float)
        self.sampler = sampler

    def build_bqm(self, hyperparams: ArrayLike = None):
        """
        Build the binary quadratic model for the problem scenario.

        Args:
            hyperparams (ArrayLike, optional): The hyperparameters.

        Returns:
            BinaryQuadraticModel: The binary quadratic model for the problem scenario.
        """
        # Asserts for input validation
        assert self.num_new_cs > 0, "Number of new charging stations must be > 0"
        assert self.num_poi > 0, "Number of POIs must be > 0"
        assert self.num_cs > 0, "Number of existing charging stations must be > 0"

        # Unpack hyperparameters
        if hyperparams is not None:
            a, b, c, d = np.asarray(hyperparams, dtype=float)
            assert len(hyperparams) == 4, "Hyperparameters must be of length 4"
        else:
            a, b, c, d = self.hyperparams

        # Compute coefficients for the objective functions
        num_nodes = len(self.potential_nodes)
        alpha = a
        beta = b
        gamma = c
        delta = d

        # Initialize the binary quadratic model
        bqm = dimod.BinaryQuadraticModel(num_nodes, "BINARY")

        # Constraint 1: Min average distance to POIs
        for i, cand_loc in enumerate(self.potential_nodes):
            dist = np.sum(influence_matrix(self.pois, [cand_loc], self.sigma))
            bqm.add_linear(i, -dist * alpha)

        # Constraint 2: Max distance to existing chargers
        for i, cand_loc in enumerate(self.potential_nodes):
            dist = np.sum(
                influence_matrix(self.charging_stations, [cand_loc], self.sigma)
            )
            bqm.add_linear(i, dist * beta)

        # Constraint 3: Max distance to other new charging locations
        for (i, ai), (j, aj) in combinations(enumerate(self.potential_nodes), 2):
            dist = np.sum(influence_matrix([ai], [aj], self.sigma))
            bqm.add_interaction(i, j, dist * gamma)

        # Constraint 4: Choose exactly num_new_cs new charging locations
        bqm.update(
            dimod.generators.combinations(
                bqm.variables, self.num_new_cs, strength=delta
            )
        )
        return bqm

    def build_bqm_numpy(self, hyperparams: ArrayLike = None):
        """
        Build the binary quadratic model for the problem scenario using numpy vectors.

        Args:
            hyperparams (ArrayLike, optional): The hyperparameters.

        Returns:
            BinaryQuadraticModel: The binary quadratic model for the problem scenario.
        """
        # Asserts for input validation
        assert self.num_new_cs > 0, "Number of new charging stations must be > 0"
        assert self.num_poi > 0, "Number of POIs must be > 0"
        assert self.num_cs > 0, "Number of existing charging stations must be > 0"

        # Unpack hyperparameters
        if hyperparams is not None:
            alpha, beta, gamma, delta = np.asarray(hyperparams, dtype=float)
            assert len(hyperparams) == 4, "Hyperparameters must be of length 4"
        else:
            alpha, beta, gamma, delta = self.hyperparams

        # Linear coefficients
        linear = self.__build_linear(alpha, beta)

        # Quadratic coefficients
        quadratic = self.__build_quadratic(gamma)

        # Create the binary quadratic model
        bqm = dimod.BinaryQuadraticModel.from_numpy_vectors(
            linear, quadratic, 0, dimod.BINARY
        )
        # Constraint 4: Choose exactly num_new_cs new charging locations
        bqm.update(
            dimod.generators.combinations(
                bqm.variables, self.num_new_cs, strength=delta
            )
        )
        return bqm

    def __build_linear(self, alpha: float, beta: float) -> np.ndarray:
        """
        Build the linear coefficients for the QUBO model.

        Args:
            alpha (float): The alpha value.
            beta (float): The beta value.

        Returns:
            np.ndarray: The linear coefficients.
        """
        # Initialize the binary quadratic model
        num_nodes = len(self.potential_nodes)
        linear = np.zeros(num_nodes)

        # Constraint 1: Min average distance to POIs
        dist_pois = influence_matrix(self.pois, self.potential_nodes, self.sigma)
        linear -= np.sum(dist_pois, axis=0) * alpha

        # Constraint 2: Max distance to existing chargers
        dist_cs = influence_matrix(
            self.charging_stations, self.potential_nodes, self.sigma
        )
        linear += np.sum(dist_cs, axis=0) * beta
        return linear

    def __build_quadratic(self, gamma: float) -> np.ndarray:
        """
        Build the quadratic coefficients for the QUBO model.

        Args:
            gamma (float): The gamma value.

        Returns:
            np.ndarray: The quadratic coefficients.
        """
        num_nodes = len(self.potential_nodes)

        # Compute the distance matrix between potential new charging locations
        dist_new_cs = gamma * influence_matrix(
            self.potential_nodes, self.potential_nodes, self.sigma
        )

        # Create quadratic coefficients
        quad_row = np.tile(np.arange(num_nodes), (num_nodes, 1)).flatten("F")
        quad_col = np.tile(np.arange(num_nodes), num_nodes)
        dist_new_cs = np.triu(dist_new_cs, 1).flatten()

        # Remove zero values from the quadratic coefficients
        q1 = quad_row[dist_new_cs != 0]
        q2 = quad_col[dist_new_cs != 0]
        q3 = dist_new_cs[dist_new_cs != 0]

        return q1, q2, q3

    def run_bqm(self, bqm, **kwargs):
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
        # Run the sampler
        sampleset = self.sampler.sample(bqm, **kwargs)
        return sampleset

    def score_sampleset(self, sampleset) -> float:
        """
        Score the sampleset based on the distance to POIs and existing charging stations.
        """
        n_samples = len(sampleset)

        # Extract the new charging nodes from the sampleset
        new_charging_nodes = [
            [self.potential_nodes[k] for k, v in sample.items() if v == 1]
            for sample in sampleset
        ]
        # Calculate the score for each sample
        score = np.zeros(n_samples)
        best_score = 0
        for i, new_cs in enumerate(new_charging_nodes):
            score[i] = self.fitness(new_cs)
            if score[i] > best_score:
                best_score = score[i]
                self.new_charging_nodes = new_cs
        total_score = np.mean(score) - np.var(score)
        return total_score

    def search_hyperparams(
        self, init_guess: ArrayLike = [2, 6, 7, 2], **kwargs
    ) -> np.ndarray:
        """
        Search for the best hyperparameters for the quantum annealing algorithm.

        Args:
            init_guess (np.array): Initial guess for the hyperparameters.

        Returns:
            np.array: Best hyperparameters found.
        """
        assert isinstance(
            self.sampler, SimulatedAnnealingSampler
        ), "Sampler must be a SimulatedAnnealingSampler"

        # Define the objective function
        def objective(hyperparams):
            print(f"Trying hyperparameters: {hyperparams}")
            bqm = self.build_bqm(hyperparams=hyperparams)
            sampleset = self.run_bqm(bqm, **kwargs)
            score = self.score_sampleset(sampleset)
            return -score

        # Optimize the hyperparameters
        init_guess = np.asarray(init_guess, dtype=float)
        best_result = minimize(objective, init_guess, method="Nelder-Mead")
        best_params = best_result.x
        return best_params


class GeneticAlgortihm(EVCP):
    """
    Genetic algorithm to solve the problem of finding the optimal locations
    for new charging stations.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        num_poi: int,
        num_cs: int,
        num_new_cs: int,
        pop_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        generations: int = 500,
        seed: Optional[int] = None,
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
        # Initialize the scenario
        super().__init__(shape, num_poi, num_cs, num_new_cs, seed=seed)

        # Set up parameters
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population: List[List[Tuple[int, int]]] = []
        self.best_fitness = 0
        self.best_generation = 0

        self.__init_population()

    def __init_population(self) -> None:
        """
        Initialize the population with random coordinates.
        """
        self.population = [
            random.choices(self.potential_nodes, k=self.num_new_cs)
            for _ in range(self.pop_size)
        ]

    def __tournament_selection(self, k: int = 2) -> List[Tuple[int, int]]:
        """
        Perform a tournament selection to select a parent.

        Args:
            k (int, optional): Number of individuals to select. Defaults to 2.

        Returns:
            List[Tuple[int, int]]: The selected parent.
        """
        selected = random.choices(self.population, k=k)
        return max(selected, key=self.fitness)

    def __crossover(
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

    def __mutate(self, child: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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
        for gen in range(self.generations):
            new_population: List[List[Tuple[int, int]]] = []

            # Select parents and perform crossovers and/or mutations
            for _ in range(self.pop_size):
                parent1 = self.__tournament_selection()
                if random.random() < self.crossover_rate:
                    parent2 = self.__tournament_selection()
                    child = self.__crossover(parent1, parent2)
                else:
                    child = parent1
                child = self.__mutate(child)
                new_population.append(child)

            # Update the population
            self.population = new_population

            # Update the best solution found so far
            for candidate in self.population:
                fit = self.fitness(candidate)
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.new_charging_nodes = candidate
                    self.best_generation = gen

        # Ensure the new charging nodes are in the correct format
        self.new_charging_nodes = [tuple(node) for node in self.new_charging_nodes]
