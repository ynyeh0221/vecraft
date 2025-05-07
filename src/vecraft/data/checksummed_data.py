import base64
import hashlib
import json
import struct
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
    """
    A class representing a data packet for storing, serializing, and deserializing data records.

    DataPacket provides data integrity validation, supports multiple serialization methods, and
    can be converted to other data structures. Each DataPacket instance contains an automatically
    calculated checksum to verify data integrity.

    Attributes:
        type (DataPacketType): The type of data packet, which can be RECORD, TOMBSTONE, or NONEXISTENT.
        record_id (str): The unique identifier for the record.
        checksum_algorithm (Union[str, ChecksumFunc]): The algorithm used to calculate the checksum,
                                                       default is 'sha256'.
        original_data (Any): The original data, which can be of any type.
        vector (Optional[np.ndarray]): Vector data, an optional numpy array.
        metadata (Optional[Dict[str, Any]]): Metadata associated with the record.
        checksum (str): Automatically calculated checksum for validating data integrity.
    """
    type: DataPacketType
    record_id: str
    checksum_algorithm: Union[str, ChecksumFunc] = 'sha256'
    original_data: Any = None
    vector: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    checksum: str = field(init=False)

    def __post_init__(self):
        """
        Automatically calculates the checksum after initialization.

        Process: Uses the specified checksum algorithm to process the serialized field data.
        """
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

    def to_bytes(self) -> bytes:
        """
        Converts the DataPacket to bytes.

        Serializes the DataPacket into a compact binary format suitable for persistent storage
        or network transmission.

        Returns:
            bytes: The serialized binary data.
        """
        record_id = self.record_id
        orig = self.original_data
        vec = self.vector
        meta = self.metadata
        checksum = self.checksum

        rid_b = record_id.encode('utf-8')
        doc = json.dumps(orig)
        orig_b = doc.encode('utf-8')
        vec_b = vec.tobytes()
        meta_b = json.dumps(meta).encode('utf-8')
        checksum_b = checksum.encode('utf-8')

        header = struct.pack('<5I', len(record_id), len(orig_b), len(vec_b), len(meta_b), len(checksum_b))
        return header + rid_b + orig_b + vec_b + meta_b + checksum_b

    @staticmethod
    def from_bytes(data: bytes) -> 'DataPacket':
        """
        Reconstructs a DataPacket from bytes.

        Deserializes from a binary format to a DataPacket instance and verifies the checksum.

        Args:
            data (bytes): Binary format DataPacket data.

        Returns:
            DataPacket: The reconstructed DataPacket instance.

        Raises:
            ChecksumValidationFailureError: If the stored checksum doesn't match the recalculated checksum.
        """
        header_size = 5 * 4
        rid_len, original_data_size, vector_size, metadata_size, checksum_size = struct.unpack(
            '<5I',
            data[:header_size]
        )

        pos = header_size
        returned_record_id_bytes = data[pos:pos + rid_len]
        returned_record_id = str(returned_record_id_bytes, 'utf-8')
        pos += rid_len
        orig_bytes = data[pos:pos + original_data_size]
        pos += original_data_size
        vec_bytes = data[pos:pos + vector_size]
        pos += vector_size
        meta_bytes = data[pos:pos + metadata_size]
        pos += metadata_size
        checksum_bytes = data[pos:pos + checksum_size]
        stored_checksum = str(checksum_bytes, 'utf-8')

        data_packet = DataPacket(type=DataPacketType.RECORD,
                                 record_id=returned_record_id,
                                 original_data=json.loads(orig_bytes.decode('utf-8')),
                                 vector=np.frombuffer(vec_bytes, dtype=np.float32),
                                 metadata=json.loads(meta_bytes.decode('utf-8')))

        if stored_checksum != data_packet.checksum:
            error_message = f"Record id {data_packet.record_id}'s stored checksum {stored_checksum} does not match reconstructed checksum {data_packet.checksum}"
            raise ChecksumValidationFailureError(error_message)

        return data_packet

    def to_index_item(self) -> 'IndexItem':
        """
        Converts the DataPacket to an IndexItem.

        Creates an IndexItem instance containing the record_id and vector, and verifies its integrity.

        Returns:
            IndexItem: IndexItem instance created from the DataPacket.

        Raises:
            ChecksumValidationFailureError: If the converted data fails checksum validation.
        """
        index_item = IndexItem(record_id=self.record_id, vector=self.vector)
        self.validate_checksum()
        reconstructed = DataPacket(type=self.type, record_id=index_item.record_id, original_data=self.original_data, vector=index_item.vector, metadata=self.metadata)
        if reconstructed.checksum != self.checksum:
            raise ChecksumValidationFailureError("Reconstruct checksum validation failed")
        index_item.validate_checksum()
        return index_item

    def to_doc_item(self) -> 'DocItem':
        """
        Converts the DataPacket to a DocItem.

        Creates a DocItem instance containing the record_id and document (original_data),
        and verifies its integrity.

        Returns:
            DocItem: DocItem instance created from the DataPacket.

        Raises:
            ChecksumValidationFailureError: If the converted data fails checksum validation.
        """
        doc_item = DocItem(record_id=self.record_id, document=self.original_data)
        self.validate_checksum()
        reconstructed = DataPacket(type=self.type, record_id=doc_item.record_id, original_data=doc_item.document, vector=self.vector, metadata=self.metadata)
        if reconstructed.checksum != self.checksum:
            raise ChecksumValidationFailureError("Reconstruct checksum validation failed")
        doc_item.validate_checksum()
        return doc_item

    def to_metadata_item(self) -> 'MetadataItem':
        """
        Converts the DataPacket to a MetadataItem.

        Creates a MetadataItem instance containing the record_id and metadata,
        and verifies its integrity.

        Returns:
            MetadataItem: MetadataItem instance created from the DataPacket.

        Raises:
            ChecksumValidationFailureError: If the converted data fails checksum validation.
        """
        meta_item = MetadataItem(record_id=self.record_id, metadata=self.metadata)
        self.validate_checksum()
        reconstructed = DataPacket(type=self.type, record_id=meta_item.record_id, original_data=self.original_data, vector=self.vector, metadata=meta_item.metadata)
        if reconstructed.checksum != self.checksum:
            raise ChecksumValidationFailureError("Reconstruct checksum validation failed")
        meta_item.validate_checksum()
        return meta_item

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

@dataclass
class LocationItem:
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
    def from_dict(cls, d: Dict[str, Any]) -> 'LocationIndex':
        """
        Reconstruct a LocationIndex from a dictionary.
        """
        # Create the LocationIndex
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