"""Data classes representing packets in the index layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from vecraft_data_model.checksum_util import get_checksum_func, _concat_bytes, _prepare_field_bytes
from vecraft_exception_model.exception import ChecksumValidationFailureError

# Define a Vector type
Vector = np.ndarray


@dataclass
class VectorPacket:
    """Container storing a vector and its identifier.

    Attributes:
        record_id: Unique ID for the vector.
        vector: Numpy array containing the vector data.
        checksum_algorithm: Algorithm used to compute ``checksum``.
        checksum: Hex encoded checksum computed in ``__post_init__``.
    """
    record_id: str
    vector: Vector
    checksum_algorithm: str = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        """Compute and store the checksum after initialization."""
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize vector_index item fields into bytes for checksum calculation.
        """
        parts: List[bytes] = [self.record_id.encode('utf-8')]

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
    """Container for an indexed document.

    Attributes:
        record_id: Unique ID for the document.
        document: Text contents for the record.
        checksum_algorithm: Algorithm used to compute ``checksum``.
        checksum: Hex encoded checksum computed in ``__post_init__``.
    """
    record_id: str
    document: str
    checksum_algorithm: str = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        """Compute and store the checksum after initialization."""
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize vector_index item fields into bytes for checksum calculation.
        """
        parts: List[bytes] = [self.record_id.encode('utf-8'), _prepare_field_bytes(self.document)]

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
    """Container for record metadata.

    Attributes:
        record_id: Unique ID for the record.
        metadata: Arbitrary key/value metadata for the record.
        checksum_algorithm: Algorithm used to compute ``checksum``.
        checksum: Hex encoded checksum computed in ``__post_init__``.
    """
    record_id: str
    metadata: Dict[str, Any]
    checksum_algorithm: str = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        """Compute and store the checksum after initialization."""
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize metadata item fields into bytes for checksum calculation.
        """
        parts: List[bytes] = [self.record_id.encode('utf-8')]

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
    """Record location information for a stored object.

    Attributes:
        record_id: ID of the record this location refers to.
        offset: Byte offset where the record begins.
        size: Size in bytes of the stored record.
        checksum_algorithm: Algorithm used to compute ``checksum``.
        checksum: Hex encoded checksum computed in ``__post_init__``.
    """
    record_id: str  # ID of the record this location refers to
    offset: int  # Starting byte position in storage
    size: int  # Size in bytes of the record
    checksum_algorithm: str = 'sha256'

    # The checksum field will be initialized in __post_init__
    checksum: str = field(init=False)

    def __post_init__(self):
        """Compute and store the checksum after initialization."""
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize location index fields into bytes for checksum calculation.
        """
        parts: List[bytes] = [self.record_id.encode('utf-8'), _prepare_field_bytes(self.offset),
                              _prepare_field_bytes(self.size)]

        # Include record_id

        # Include offset and size

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
        checksum_algorithm (str | ChecksumFunc): The algorithm used to calculate the checksum,
                                                      default is 'sha256'.
        checksum (str): Automatically calculated checksum for validating data integrity.
    """
    name: str
    dim: int
    vector_type: str
    checksum_algorithm: str = 'sha256'

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
        parts: List[bytes] = [self.name.encode('utf-8'), _prepare_field_bytes(self.dim),
                              self.vector_type.encode('utf-8')]

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
