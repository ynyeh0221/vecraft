import hashlib
import json
import zlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Callable, Union, List

import numpy as np

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


@dataclass
class DataPacket:
    type: str
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

        parts.append(self.type.encode('utf-8'))
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
        Recompute checksum and compare. Returns False if data was corrupted.
        """
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        return func(raw) == self.checksum

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['vector'] = self.vector.tolist() if self.vector is not None else []
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataPacket':
        packet = cls(
            type=d['type'],
            original_data=d['original_data'],
            vector=np.array(d['vector']),
            metadata=d['metadata'],
            record_id=d.get('record_id'),
            checksum_algorithm=d.get('checksum_algorithm', 'sha256')
        )
        packet.checksum = d['checksum']
        return packet


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
        parts: List[bytes] = [self.query_vector.tobytes(), _prepare_field_bytes(self.k),
                              _prepare_field_bytes(self.where or {}), _prepare_field_bytes(self.where_document or {})]
        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        return func(raw) == self.checksum

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_vector': self.query_vector.tolist(),
            'k': self.k,
            'where': self.where,
            'where_document': self.where_document,
            'checksum_algorithm': self.checksum_algorithm,
            'checksum': self.checksum,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QueryPacket':
        packet = cls(
            query_vector=np.array(d['query_vector']),
            k=d['k'],
            where=d.get('where'),
            where_document=d.get('where_document'),
            checksum_algorithm=d.get('checksum_algorithm', 'sha256')
        )
        packet.checksum = d['checksum']
        return packet

# Define Vector type
Vector = np.ndarray


@dataclass
class IndexItem:
    """Vector with associated ID, document content, and metadata."""
    record_id: str
    vector: Vector
    document: Optional[str] = None  # Original document content
    metadata: Optional[Dict[str, Any]] = None
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        # compute checksum from serialized fields
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """
        Serialize index item fields into bytes for checksum calculation.
        """
        parts: List[bytes] = []

        parts.append(self.record_id.encode('utf-8'))

        if self.vector is not None:
            parts.append(self.vector.tobytes())
        else:
            parts.append(b"null")

        if self.document is not None:
            parts.append(self.document.encode('utf-8'))
        else:
            parts.append(b"null")

        if self.metadata is not None:
            parts.append(_prepare_field_bytes(self.metadata))
        else:
            parts.append(b"null")

        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        """
        Recompute checksum and compare. Returns False if data was corrupted.
        """
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        return func(raw) == self.checksum


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
        Recompute checksum and compare. Returns False if data was corrupted.
        """
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        return func(raw) == self.checksum


def validate_checksum(func):
    """
    A decorator that validates parameters of specific types:
    IndexItem, DataPacket, QueryPacket, or MetadataItem.
    """

    def wrapper(self, *args, **kwargs):
        # Define the types to validate
        types_to_validate = (IndexItem, DataPacket, QueryPacket, MetadataItem)

        # Collect all parameters of the specified types
        items_to_validate = []

        # Check positional arguments
        for arg in args:
            if isinstance(arg, types_to_validate):
                items_to_validate.append(arg)

        # Check keyword arguments
        for arg in kwargs.values():
            if isinstance(arg, types_to_validate):
                items_to_validate.append(arg)

        # Pre-validation
        for item in items_to_validate:
            item.validate_checksum()

        # Execute the original function
        result = func(self, *args, **kwargs)

        # Post-validation
        for item in items_to_validate:
            item.validate_checksum()

        return result

    return wrapper