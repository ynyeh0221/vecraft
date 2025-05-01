
from dataclasses import dataclass

@dataclass
class Field:
    name: str
    dim: int
    vector_type: str

@dataclass
class CollectionSchema:
    name: str
    field: Field

