
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class VectorType(ABC):
    """Vector type plugin interface."""

    @abstractmethod
    def encode(self, raw: Any) -> np.ndarray:
        ...

    @abstractmethod
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        ...
