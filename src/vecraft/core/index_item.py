from dataclasses import dataclass

import numpy as np

# Define Vector type
Vector = np.ndarray

@dataclass
class IndexItem:
    """Represents a vector with its associated record ID."""
    id: str
    vector: Vector