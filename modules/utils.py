import os
from typing import List, Tuple, Optional, Union

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


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


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculate the distance between two points on a grid.
    """
    # return cityblock(a, b)
    return (a[0] ** 2 - 2 * a[0] * b[0] + b[0] ** 2) + (
        a[1] ** 2 - 2 * a[1] * b[1] + b[1] ** 2
    )


def influence_matrix(
    x: Tuple[int, int], y: Tuple[int, int], sigma: Union[float, int] = 1
):
    """
    Calculate the influence matrix between two sets of coordinates.

    Args:
        x (Tuple[int, int]): The first set of coordinates.
        y (Tuple[int, int]): The second set of coordinates.
        sigma (Union[float, int], optional): The sigma value. Defaults to 1.

    Returns:
        np.ndarray: The influence matrix.
    """
    dist_matrix = cdist(x, y, "cityblock")
    return np.exp(-(dist_matrix**2) / (2 * sigma**2))
