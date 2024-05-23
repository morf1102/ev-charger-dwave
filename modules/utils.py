import os
from typing import List, Tuple, Optional

import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist, cityblock
import matplotlib.pyplot as plt


def load_data(folder_path: str) -> gpd.GeoDataFrame:
    """
    Load all geojson files in the folder_path and return a GeoDataFrame with all the data

    Args:
        folder_path (str): Path to the folder containing the geojson files

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing all the data from the geojson files
    """
    gdf = []
    for file in os.listdir(folder_path):
        if file.endswith(".geojson") and file != "lightposts_curated.geojson":
            if folder_path[-1] != "/":
                folder_path += "/"
            temp = gpd.read_file(folder_path + file).drop(columns=["description", "id"])
            gdf.append(temp)

    gdf = pd.concat(gdf)
    return gdf


def output_image(
    graph: nx.Graph,
    pois: List[Tuple[int, int]],
    charging_stations: List[Tuple[int, int]],
    new_charging_nodes: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """Create output image of solution scenario.

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
    # fig.suptitle("New EV Charger Locations")
    pos = {x: [x[0], x[1]] for x in graph.nodes()}

    # Locate POIs in map
    poi_graph = graph.subgraph(pois)

    # Locate old charging stations in map
    cs_graph = graph.subgraph(charging_stations)

    nx.draw_networkx(
        graph,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_color="k",
        node_size=3,
    )
    nx.draw_networkx(
        poi_graph,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_color="b",
        node_size=75,
    )
    nx.draw_networkx(
        cs_graph,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_color="r",
        node_size=75,
    )
    if new_charging_nodes is not None:
        if isinstance(new_charging_nodes[0], list):
            new_charging_nodes = [tuple(x) for x in new_charging_nodes]
        new_cs_graph = graph.subgraph(new_charging_nodes)
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


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculate the distance between two points on a grid.
    """
    #return cityblock(a, b)
    return (a[0] ** 2 - 2 * a[0] * b[0] + b[0] ** 2) + (
        a[1] ** 2 - 2 * a[1] * b[1] + b[1] ** 2
    )


def score(pois, charging_stations, new_charging_nodes):
    """
    Calculate the score of the solution scenario.
    """
    cs = charging_stations + new_charging_nodes
    sigma = 1

    dist_matrix = cdist(pois, cs, distance)

    inflence_matrix = np.exp(-dist_matrix / (2 * sigma**2))
    total_influence = np.sum(inflence_matrix, axis=1)
    return np.sum(total_influence)

    """cs = charging_stations + new_charging_nodes
    weights = np.linspace(0, 1, len(cs))

    dist_matrix = cdist(pois, cs, distance)

    coverage = np.sum(np.min(dist_matrix <= 10, axis=1))
    access = np.sum(np.sort(1 / (dist_matrix + 1e-5), axis=1) * weights)

    print(f"Coverage: {coverage}")
    print(f"Access: {access}")

    return coverage"""
