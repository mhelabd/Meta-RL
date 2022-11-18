import numpy as np


def dist(xpos1, xpos2):
    """Return distances between two xpos arrays.
    Arrays must be same shape of (n,3) or be broadcastable."""
    return np.linalg.norm(xpos1 - xpos2, axis=1)
