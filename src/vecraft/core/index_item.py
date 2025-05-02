from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

# Define Vector type
Vector = np.ndarray

@dataclass
class IndexItem:
    """Vector with associated ID, document content, and metadata."""
    id: str
    vector: Vector
    document: Optional[str] = None  # Original document content
    metadata: Optional[Dict[str, Any]] = None