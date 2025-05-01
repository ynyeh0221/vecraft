
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

Vector = np.ndarray

class Index(ABC):
    """Abstract vector index."""

    @abstractmethod
    def build(self, vectors: Vector) -> None:
        ...

    @abstractmethod
    def add(self, vec: Vector, id_: int) -> None:
        ...

    @abstractmethod
    def search(self, query: Vector, k: int) -> List[Tuple[int, float]]:
        ...
