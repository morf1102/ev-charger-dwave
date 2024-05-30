import random
from typing import List, Tuple, Optional

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from .utils import influence_matrix


class EVCP:
    """
    Class to set up and visualize a scenario for electric vehicle charging station placement.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        num_poi: int,
        num_cs: int,
        num_new_cs: int,
        seed: Optional[int] = None,
    ):
        """
        Initialize the scenario for electric vehicle charging station placement.

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
            num_poi (int): The number of points of interest.
            num_cs (int): The number of existing charging stations.
            num_new_cs (int): The number of new charging stations to place.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        # Set random seed for reproducibility
        random.seed(seed)

        self.width, self.height = shape
        self.num_cs = num_cs
        self.num_new_cs = num_new_cs
        self.num_poi = num_poi

        self.sigma = 1
        self.pois = []
        self.charging_stations = []
        self.potential_nodes = []
        self.new_charging_nodes = []

        self.__set_up_scenario()
        self.__get_sigma()

    def __set_up_scenario(self) -> None:
        """
        Build scenario set up with specified parameters.

        Args:
            width (int): Width of grid.
            height (int): Height of grid.
            num_poi (int): Number of points of interest.
            num_cs (int): Number of existing charging stations.
        """

        # Create a grid graph
        self.graph = nx.grid_2d_graph(self.width, self.height)
        nodes = list(self.graph.nodes)

        # Identify a fixed set of points of interest and charging locations
        self.pois = random.sample(nodes, k=self.num_poi)
        self.charging_stations = random.sample(nodes, k=self.num_cs)

        # Identify potential new charging locations
        self.potential_nodes = list(
            self.graph.nodes() - set(self.charging_stations) - set(self.pois)
        )

    def draw_grid(self) -> plt.Figure:
        """
        Create output image of solution scenario.

        Args:
            G (networkx graph): Grid graph of size w by h
            pois (list of tuples of ints): A fixed set of points of interest
            charging_stations (list of tuples of ints):
                A fixed set of current charging locations
            new_charging_nodes (list of tuples of ints):
                Locations of new charging stations

        Returns:
            None. Output saved to file "map.png".
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis("off")

        pos = {x: x for x in self.graph.nodes()}

        # Locate POIs and existing charging stations in map
        poi_graph = self.graph.subgraph(self.pois)
        cs_graph = self.graph.subgraph(self.charging_stations)

        # Draw the grid graph
        nx.draw_networkx(
            self.graph, ax=ax, pos=pos, with_labels=False, node_color="k", node_size=3
        )
        nx.draw_networkx(
            poi_graph, ax=ax, pos=pos, with_labels=False, node_color="b", node_size=75
        )
        nx.draw_networkx(
            cs_graph, ax=ax, pos=pos, with_labels=False, node_color="r", node_size=75
        )
        if len(self.new_charging_nodes) > 0:
            if isinstance(self.new_charging_nodes[0], list):
                self.new_charging_nodes = [tuple(x) for x in self.new_charging_nodes]
            new_cs_graph = self.graph.subgraph(self.new_charging_nodes)
            nx.draw_networkx(
                new_cs_graph,
                ax=ax,
                pos=pos,
                with_labels=False,
                node_color="g",
                node_size=75,
            )

        plt.close()
        return fig

    def __get_sigma(self) -> float:
        """
        Calculate the sigma value for the fitness function.

        Returns:
            float: The sigma value.
        """
        avg_dist = np.mean(cdist(self.pois, self.charging_stations, "cityblock"))
        self.sigma = avg_dist / np.sqrt(2 * np.log(1000))

    def fitness(self, new_charging_nodes: List[Tuple[int, int]]) -> float:
        """
        Calculate the fitness of the solution scenario.

        Args:
            new_charging_nodes (List[Tuple[int, int]]): The new charging locations.

        Returns:
            float: The fitness of the solution.
        """
        cs = np.append(self.charging_stations, new_charging_nodes, axis=0)

        pois_influence_matrix = influence_matrix(self.pois, cs, self.sigma)
        cs_influence_matrix = influence_matrix(cs, new_charging_nodes, self.sigma)

        total_influence = (
            100 * np.sum(pois_influence_matrix) / np.sum(cs_influence_matrix)
        )
        return total_influence
