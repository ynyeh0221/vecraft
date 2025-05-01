
import numpy as np
from src.core.vector_type_interface import VectorType

class ImageVector(VectorType):
    def encode(self, raw) -> np.ndarray:
        # TODO: actual image encoding
        return np.zeros(512, dtype=np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return 0.0
