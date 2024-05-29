from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def influence_matrix(
    x: List[Tuple[int, int]], y: List[Tuple[int, int]], sigma: float = 1.
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
