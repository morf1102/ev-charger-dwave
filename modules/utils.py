import os
from typing import List, Tuple, Optional, Union

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculate the distance between two points on a grid.
    """
    # return cityblock(a, b)
    return (a[0] ** 2 - 2 * a[0] * b[0] + b[0] ** 2) + (
        a[1] ** 2 - 2 * a[1] * b[1] + b[1] ** 2
    )


def influence_matrix(
    x: List[Tuple[int, int]], y: List[Tuple[int, int]], sigma: Union[float, int] = 1
):
    """
    Calculate the gaussian influence matrix between two sets of coordinates.

    Args:
        x (Tuple[int, int]): The first set of coordinates.
        y (Tuple[int, int]): The second set of coordinates.
        sigma (Union[float, int], optional): The sigma value. Defaults to 1.

    Returns:
        np.ndarray: The gaussian influence matrix.
    """
    dist_matrix = cdist(x, y, "cityblock")
    return np.exp(-np.divide(np.square(dist_matrix), 2 * sigma**2))
