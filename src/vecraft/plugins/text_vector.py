
import numpy as np
from src.vecraft.core.vector_type_interface import VectorType

class TextVector(VectorType):
    def encode(self, raw: str) -> np.ndarray:
        # TODO: actual encoding
        return np.zeros(768, dtype=np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # TODO: compute cosine similarity
        return 0.0
