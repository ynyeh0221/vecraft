import base64
import hashlib
import json
import zlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional, Callable, Union, List

import numpy as np

from src.vecraft.data.exception import ChecksumValidationFailureError

# Type for checksum function: takes bytes -> hex string
ChecksumFunc = Callable[[bytes], str]


# Predefined checksum functions
_PREDEFINED_CHECKSUMS: Dict[str, ChecksumFunc] = {
    'crc32': lambda b: format(zlib.crc32(b) & 0xFFFFFFFF, '08x'),
    'md5': lambda b: hashlib.md5(b).hexdigest(),
    'sha1': lambda b: hashlib.sha1(b).hexdigest(),
    'sha256': lambda b: hashlib.sha256(b).hexdigest(),
}


def get_checksum_func(
    algorithm: Union[str, ChecksumFunc]
) -> ChecksumFunc:
    """
    Resolve an algorithm name or accept a custom function.
    """
    if callable(algorithm):
        return algorithm
    alg = algorithm.lower()
    if alg in _PREDEFINED_CHECKSUMS:
        return _PREDEFINED_CHECKSUMS[alg]
    # fallback to hashlib
    try:
        def _fn(b: bytes, name=alg):
            h = hashlib.new(name)
            h.update(b)
            return h.hexdigest()
        _ = _fn(b"test")
        return _fn
    except Exception:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")


def _prepare_field_bytes(value: Any) -> bytes:
    """
    Prepare any Python value into bytes for checksum.
    Uses JSON (sorted keys) when possible, else repr().
    """
    try:
        return json.dumps(value, sort_keys=True).encode('utf-8')
    except Exception:
        return repr(value).encode('utf-8')


def _concat_bytes(components: List[bytes]) -> bytes:
    """
    Concatenate multiple byte components deterministically.
    """
    return b"".join(components)

class DataPacketType(Enum):
    RECORD = 1
    TOMBSTONE = 2
    NONEXISTENT = 3

@dataclass
class DataPacket:
    type: DataPacketType
    record_id: str
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'
    original_data: Any = None
    vector: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    checksum: str = field(init=False)

    def __post_init__(self):
        # compute checksum from serialized fields
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize packet fields into bytes for checksum calculation.
        """
        parts: List[bytes] = []

        parts.append(self.type.name.encode('utf-8'))
        parts.append(self.record_id.encode('utf-8'))

        # Handle fields which might be None
        if self.original_data is not None:
            parts.append(_prepare_field_bytes(self.original_data))
        else:
            parts.append(b"null")

        if self.vector is not None:
            parts.append(self.vector.tobytes())
        else:
            parts.append(b"null")

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
            raise ChecksumValidationFailureError("DataPacket checksum validation failed",
                                                 record_id=self.record_id)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Turn this DataPacket into a JSON‐friendly dict, preserving the
        exact numpy array bytes via base64.
        """
        d = asdict(self)

        # Handle enum fields by converting them to their names
        for key, value in d.items():
            if isinstance(value, Enum):
                d[key] = value.name

        # encode vector bytes, dtype and shape
        if self.vector is not None:
            v_bytes = self.vector.tobytes()
            d['vector'] = {
                'b64': base64.b64encode(v_bytes).decode('ascii'),
                'dtype': str(self.vector.dtype),
                'shape': self.vector.shape
            }
        else:
            d['vector'] = None

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataPacket':
        """
        Reconstruct a DataPacket **exactly** as it was, including
        the original vector bytes so that checksums still match.
        """
        vec_info = d.get('vector')
        if vec_info is not None:
            # decode the exact bytes back into a ndarray
            raw = base64.b64decode(vec_info['b64'].encode('ascii'))
            dtype = np.dtype(vec_info['dtype'])
            shape = tuple(vec_info['shape'])
            vector = np.frombuffer(raw, dtype=dtype).reshape(shape)
        else:
            vector = None

        # Convert string back to enum for the 'type' field
        packet_type = DataPacketType[d['type']] if isinstance(d['type'], str) else d['type']

        packet = cls(
            type=packet_type,
            record_id=d['record_id'],
            checksum_algorithm=d.get('checksum_algorithm', 'sha256'),
            original_data=d.get('original_data'),
            vector=vector,
            metadata=d.get('metadata')
        )
        # restore the original checksum (skip re‐hashing)
        packet.checksum = d['checksum']
        return packet

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        # Compare vectors using np.array_equal
        vectors_equal = ((self.vector is None and other.vector is None) or
                         (self.vector is not None and other.vector is not None and
                          np.array_equal(self.vector, other.vector)))

        return (self.type == other.type and
                self.record_id == other.record_id and
                self.checksum_algorithm == other.checksum_algorithm and
                self.original_data == other.original_data and
                vectors_equal and
                self.metadata == other.metadata and
                self.checksum == other.checksum)


@dataclass
class SearchDataPacket:
    """
    A data structure to represent search results, containing a DataPacket
    and its distance to the query vector.
    """
    data_packet: DataPacket
    distance: float
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
        Serialize packet fields into bytes for checksum calculation.
        """
        parts: List[bytes] = []

        # Include data_packet's checksum
        parts.append(self.data_packet.checksum.encode('utf-8'))

        # Include distance
        parts.append(_prepare_field_bytes(self.distance))

        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        """
        Recompute checksum and compare. Raises ChecksumValidationFailureError if data was corrupted.
        """
        # First validate the contained DataPacket
        self.data_packet.validate_checksum()

        # Then validate this SearchDataPacket
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        if func(raw) != self.checksum:
            raise ChecksumValidationFailureError(
                "SearchDataPacket checksum validation failed",
                record_id=self.data_packet.record_id
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Turn this SearchDataPacket into a JSON-friendly dict.
        """
        d = {
            'data_packet': self.data_packet.to_dict(),
            'distance': self.distance,
            'checksum_algorithm': self.checksum_algorithm,
            'checksum': self.checksum
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SearchDataPacket':
        """
        Reconstruct a SearchDataPacket from a dictionary.
        """
        # Reconstruct the DataPacket
        data_packet = DataPacket.from_dict(d['data_packet'])

        # Create the SearchDataPacket
        packet = cls(
            data_packet=data_packet,
            distance=d['distance'],
            checksum_algorithm=d.get('checksum_algorithm', 'sha256')
        )

        # Restore the original checksum (skip re-hashing)
        packet.checksum = d['checksum']
        return packet

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return (self.data_packet == other.data_packet and
                self.distance == other.distance and
                self.checksum_algorithm == other.checksum_algorithm and
                self.checksum == other.checksum)

@dataclass
class QueryPacket:
    query_vector: np.ndarray
    k: int
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        parts: List[bytes] = [
            self.query_vector.tobytes(),
            _prepare_field_bytes(self.k),
            _prepare_field_bytes(self.where or {}),
            _prepare_field_bytes(self.where_document or {})
        ]
        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        if func(raw) != self.checksum:
            raise ChecksumValidationFailureError(
                "QueryPacket checksum validation failed"
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON‐friendly serialization that keeps vector bytes intact via base64.
        """
        # encode the exact bytes of the vector
        v_bytes = self.query_vector.tobytes()
        return {
            'query_vector': {
                'b64': base64.b64encode(v_bytes).decode('ascii'),
                'dtype': str(self.query_vector.dtype),
                'shape': self.query_vector.shape
            },
            'k': self.k,
            'where': self.where,
            'where_document': self.where_document,
            'checksum_algorithm': self.checksum_algorithm,
            'checksum': self.checksum,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QueryPacket':
        """
        Reconstruct exactly the same QueryPacket, including bit‐identical vector.
        """
        vinfo = d['query_vector']
        raw = base64.b64decode(vinfo['b64'].encode('ascii'))
        dtype = np.dtype(vinfo['dtype'])
        shape = tuple(vinfo['shape'])
        vector = np.frombuffer(raw, dtype=dtype).reshape(shape)

        packet = cls(
            query_vector=vector,
            k=d['k'],
            where=d.get('where'),
            where_document=d.get('where_document'),
            checksum_algorithm=d.get('checksum_algorithm', 'sha256')
        )
        # skip recomputing—restore original
        packet.checksum = d['checksum']
        return packet

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        # Compare query vectors using np.array_equal
        vectors_equal = np.array_equal(self.query_vector, other.query_vector)

        return (vectors_equal and
                self.k == other.k and
                self.where == other.where and
                self.where_document == other.where_document and
                self.checksum_algorithm == other.checksum_algorithm and
                self.checksum == other.checksum)

# Define Vector type
Vector = np.ndarray


@dataclass
class IndexItem:
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
class DocItem:
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
class MetadataItem:
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