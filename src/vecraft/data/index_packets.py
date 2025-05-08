from dataclasses import dataclass, field
from typing import Any, Dict, Union, List

import numpy as np

from src.vecraft.data.exception import ChecksumValidationFailureError
from src.vecraft.data.checksum_util import ChecksumFunc, get_checksum_func, _concat_bytes, _prepare_field_bytes

# Define Vector type
Vector = np.ndarray


@dataclass
class VectorPacket:
    """Vector with associated ID, document content, and metadata."""
    record_id: str
    vector: Vector
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        # compute checksum from serialized fields
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize vector_index item fields into bytes for checksum calculation.
        """
        parts: List[bytes] = []

        parts.append(self.record_id.encode('utf-8'))

        if self.vector is not None:
            parts.append(self.vector.tobytes())
        else:
            parts.append(b"null")

        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        """
        Recompute checksum and compare. Raises ChecksumValidationFailureError if data was corrupted.
        """
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        if func(raw) != self.checksum:
            raise ChecksumValidationFailureError("IndexItem checksum validation failed",
                                                 record_id=self.record_id)
        return True

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        # Compare vectors using np.array_equal
        vectors_equal = ((self.vector is None and other.vector is None) or
                         (self.vector is not None and other.vector is not None and
                          np.array_equal(self.vector, other.vector)))

        return (self.record_id == other.record_id and
                vectors_equal and
                self.checksum_algorithm == other.checksum_algorithm and
                self.checksum == other.checksum)

@dataclass
class DocumentPacket:
    """A wrapper for record ID and its associated document content."""
    record_id: str
    document: str
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        # compute checksum from serialized fields
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize vector_index item fields into bytes for checksum calculation.
        """
        parts: List[bytes] = []

        parts.append(self.record_id.encode('utf-8'))

        parts.append(_prepare_field_bytes(self.document))

        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        """
        Recompute checksum and compare. Raises ChecksumValidationFailureError if data was corrupted.
        """
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        if func(raw) != self.checksum:
            raise ChecksumValidationFailureError("IndexItem checksum validation failed",
                                                 record_id=self.record_id)
        return True

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return (self.record_id == other.record_id and
                self.document == other.document and
                self.checksum_algorithm == other.checksum_algorithm and
                self.checksum == other.checksum)


@dataclass
class MetadataPacket:
    """A wrapper for record ID and its associated metadata."""
    record_id: str
    metadata: Dict[str, Any]
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        # compute checksum from serialized fields
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize metadata item fields into bytes for checksum calculation.
        """
        parts: List[bytes] = []

        parts.append(self.record_id.encode('utf-8'))

        if self.metadata is not None:
            parts.append(_prepare_field_bytes(self.metadata))
        else:
            parts.append(b"null")

        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        """
        Recompute checksum and compare. Raises ChecksumValidationFailureError if data was corrupted.
        """
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        if func(raw) != self.checksum:
            raise ChecksumValidationFailureError("MetadataItem checksum validation failed",
                                                 record_id=self.record_id)
        return True

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return (self.record_id == other.record_id and
                self.metadata == other.metadata and
                self.checksum_algorithm == other.checksum_algorithm and
                self.checksum == other.checksum)

@dataclass
class LocationPacket:
    """
    A data structure to track the physical location of a record in storage,
    with checksum validation capabilities.
    """
    record_id: str  # ID of the record this location refers to
    offset: int  # Starting byte position in storage
    size: int  # Size in bytes of the record
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'

    # The checksum field will be initialized in __post_init__
    checksum: str = field(init=False)

    def __post_init__(self):
        # Compute checksum from serialized fields
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize location index fields into bytes for checksum calculation.
        """
        parts: List[bytes] = []

        # Include record_id
        parts.append(self.record_id.encode('utf-8'))

        # Include offset and size
        parts.append(_prepare_field_bytes(self.offset))
        parts.append(_prepare_field_bytes(self.size))

        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        """
        Recompute checksum and compare. Raises ChecksumValidationFailureError if data was corrupted.
        """
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        if func(raw) != self.checksum:
            raise ChecksumValidationFailureError(
                "LocationIndex checksum validation failed",
                record_id=self.record_id
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Turn this LocationIndex into a JSON-friendly dict.
        """
        return {
            'record_id': self.record_id,
            'offset': self.offset,
            'size': self.size,
            'checksum_algorithm': self.checksum_algorithm,
            'checksum': self.checksum
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LocationPacket':
        """
        Reconstruct a LocationPacket from a dictionary.
        """
        # Create the LocationPacket
        location = cls(
            record_id=d['record_id'],
            offset=d['offset'],
            size=d['size'],
            checksum_algorithm=d.get('checksum_algorithm', 'sha256')
        )

        # Restore the original checksum (skip re-hashing)
        if 'checksum' in d:
            location.checksum = d['checksum']

        return location

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return (self.record_id == other.record_id and
                self.offset == other.offset and
                self.size == other.size and
                self.checksum_algorithm == other.checksum_algorithm and
                self.checksum == other.checksum)

@dataclass
class CollectionSchema:
    """
    A class representing the schema for a vector collection.

    Attributes:
        name (str): The name of the collection.
        dim (int): The dimensionality of vectors in the collection.
        vector_type (str): The type of vectors stored in the collection.
        checksum_algorithm (Union[str, ChecksumFunc]): The algorithm used to calculate the checksum,
                                                      default is 'sha256'.
        checksum (str): Automatically calculated checksum for validating data integrity.
    """
    name: str
    dim: int
    vector_type: str
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        """
        Automatically calculates the checksum after initialization.
        """
        # Compute checksum from serialized fields
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize schema fields into bytes for checksum calculation.
        """
        parts: List[bytes] = []

        parts.append(self.name.encode('utf-8'))
        parts.append(_prepare_field_bytes(self.dim))
        parts.append(self.vector_type.encode('utf-8'))

        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        """
        Recompute checksum and compare. Raises ChecksumValidationFailureError if data was corrupted.
        """
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        if func(raw) != self.checksum:
            raise ChecksumValidationFailureError("CollectionSchema checksum validation failed", collection=self.name)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Turn this CollectionSchema into a JSON-friendly dict.
        """
        return {
            'name': self.name,
            'dim': self.dim,
            'vector_type': self.vector_type,
            'checksum_algorithm': self.checksum_algorithm,
            'checksum': self.checksum
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CollectionSchema':
        """
        Reconstruct a CollectionSchema from a dictionary.
        """
        schema = cls(
            name=d['name'],
            dim=d['dim'],
            vector_type=d['vector_type'],
            checksum_algorithm=d.get('checksum_algorithm', 'sha256')
        )

        # Restore the original checksum (skip re-hashing)
        if 'checksum' in d:
            schema.checksum = d['checksum']

        return schema

    def __eq__(self, other):
        """
        Compares two CollectionSchema instances for equality.
        """
        if not isinstance(other, type(self)):
            return False

        return (self.name == other.name and
                self.dim == other.dim and
                self.vector_type == other.vector_type and
                self.checksum_algorithm == other.checksum_algorithm and
                self.checksum == other.checksum)