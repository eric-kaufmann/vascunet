import numpy as np

import numpy as np

def pick_n_random_indices(arr, n):
    """
    Randomly selects n indices from the given array.

    Parameters:
    arr (array-like): The input array.
    n (int): The number of indices to select.

    Returns:
    numpy.ndarray: An array of n randomly selected indices from the input array.
    """
    return np.random.choice(np.arange(len(arr)-1), n, replace=False)

def move_and_rescale_matrix(matrix):
    """
    Move and rescale the given matrix.

    Parameters:
    matrix (numpy.ndarray): The input matrix.

    Returns:
    numpy.ndarray: The moved and rescaled matrix.
    """
    # Move the matrix to have minimum values at 0
    min_vals = np.min(matrix, axis=0)
    matrix -= min_vals

    # Rescale the matrix to have maximum values at 1
    max_vals = np.max(matrix, axis=0)
    matrix /= max_vals

    return matrix