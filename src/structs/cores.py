from __future__ import annotations

from typing import Union, Optional, List
import numpy as np

class SparseIndex:
    """
    Helper class to store index arrays for a single attribute
    indexes are in a list
    uniques are in a np.array
    """
    def __init__(self, dtype : Optional[Union[type, str]] = None):
        self.index = []
        self.uniques = np.array([])
        self.info = {}

        if dtype is not None:
            self.uniques = self.uniques.astype(dtype)

    def is_empty(self):
        return len(self.uniques) == 0

    def getv(self, loc : Union[None, int, List[int], np.ndarray] = None) -> np.ndarray:
        """Get the unique values corresponding point location"""

        if isinstance(loc, int):
            loc = [loc]

        indexer = np.asarray(self.index)
        if loc is not None:
            indexer = indexer[loc]

        # Multi-dimensional uniques: first dimension is index
        if len(self.uniques.shape) > 1:
            return self.uniques[indexer, ...]

        return self.uniques[indexer]

class DenseValue:
    def __init__(self, dtype = np.float64):
        self.dtype = np.dtype(dtype)
        self.values = np.array([], dtype=self.dtype)

    def copy_from(self, other : DenseValue,
                  loc : Optional[Union[List[int], int]] = None):
        """Copy values from another DenseValue, use loc to select a subset"""
        if loc is not None and isinstance(loc, int):
            loc = [loc]
        if loc is not None:
            self.values = np.asarray(other.values, dtype=self.dtype)[loc]
        else:
            self.values = other.values.copy()
