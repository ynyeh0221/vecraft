import base64
import json
import struct
from dataclasses import asdict, field, dataclass
from enum import Enum
from typing import Any, Dict, List, Union, Optional

import numpy as np

from src.vecraft.data.index_packets import DocumentPacket, VectorPacket, _prepare_field_bytes, _concat_bytes, \
    get_checksum_func, ChecksumFunc, MetadataPacket
from src.vecraft.data.exception import ChecksumValidationFailureError, InvalidDataException


class DataPacketType(Enum):
    RECORD = 1
    TOMBSTONE = 2
    NONEXISTENT = 3

# Add constants at module level
MAGIC_BYTES = b'VCRD'  # VeCraft Record Data
FORMAT_VERSION = 1

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
        Also validates field combinations based on packet type.

        Process:
        1. Validates that required fields are present for the packet type
        2. Uses the specified checksum algorithm to process the serialized field data

        Raises:
            ValueError: If the field combination doesn't match the packet type requirements
        """
        # Validate field combinations based on type
        if self.type == DataPacketType.RECORD:
            # For RECORD type: vector must be non-null, and at least one of metadata/document must be non-null
            if self.vector is None:
                raise InvalidDataException("RECORD type DataPacket must have a non-null vector")

            if self.original_data is None and self.metadata is None:
                raise InvalidDataException("RECORD type DataPacket must have at least one of original_data or metadata")

        elif self.type in (DataPacketType.TOMBSTONE, DataPacketType.NONEXISTENT):
            # For TOMBSTONE or NONEXISTENT: vector, metadata, and document must all be null
            if self.vector is not None:
                raise InvalidDataException(f"{self.type.name} type DataPacket must have a null vector")

            if self.original_data is not None:
                raise InvalidDataException(f"{self.type.name} type DataPacket must have a null original_data")

            if self.metadata is not None:
                raise InvalidDataException(f"{self.type.name} type DataPacket must have a null metadata")

        # Compute checksum from serialized fields (original functionality)
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
                'shape': list(self.vector.shape)  # Convert tuple to list for JSON compatibility
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

            # Handle both list and tuple for shape
            shape_data = vec_info['shape']
            shape = tuple(shape_data) if isinstance(shape_data, list) else shape_data

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
        Converts the DataPacket to bytes with magic header and version.

        Format:
        - 4 bytes: magic bytes ('VCRD')
        - 1 byte: format version
        - Original format continues...
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

        header = struct.pack('<4s B 5I',
                             MAGIC_BYTES,
                             FORMAT_VERSION,
                             len(record_id),
                             len(orig_b),
                             len(vec_b),
                             len(meta_b),
                             len(checksum_b))
        return header + rid_b + orig_b + vec_b + meta_b + checksum_b

    @staticmethod
    def from_bytes(data: bytes) -> 'DataPacket':
        """
        Reconstructs a DataPacket from bytes, validating magic header and version.
        """
        header_size = 4 + 1 + 5 * 4  # magic + version + 5 integers
        if len(data) < header_size:
            raise ValueError("Invalid data packet: too short")

        magic, version, rid_len, original_data_size, vector_size, metadata_size, checksum_size = struct.unpack(
            '<4s B 5I',
            data[:header_size]
        )

        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid magic bytes: expected {MAGIC_BYTES}, got {magic}")

        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported format version: {version}")

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

    def to_vector_packet(self) -> 'VectorPacket':
        """
        Converts the DataPacket to a VectorPacket.

        Creates a VectorPacket instance containing the record_id and vector, and verifies its integrity.

        Returns:
            VectorPacket instance created from the DataPacket.

        Raises:
            ChecksumValidationFailureError: If the converted data fails checksum validation.
        """
        index_item = VectorPacket(record_id=self.record_id, vector=self.vector)
        self.validate_checksum()
        reconstructed = DataPacket(type=self.type, record_id=index_item.record_id, original_data=self.original_data, vector=index_item.vector, metadata=self.metadata)
        if reconstructed.checksum != self.checksum:
            raise ChecksumValidationFailureError("Reconstruct checksum validation failed")
        index_item.validate_checksum()
        return index_item

    def to_document_packet(self) -> 'DocumentPacket':
        """
        Converts the DataPacket to a DocumentPacket.

        Creates a DocumentPacket instance containing the record_id and document (original_data),
        and verifies its integrity.

        Returns:
            DocumentPacket instance created from the DataPacket.

        Raises:
            ChecksumValidationFailureError: If the converted data fails checksum validation.
        """
        doc_item = DocumentPacket(record_id=self.record_id, document=self.original_data)
        self.validate_checksum()
        reconstructed = DataPacket(type=self.type, record_id=doc_item.record_id, original_data=doc_item.document, vector=self.vector, metadata=self.metadata)
        if reconstructed.checksum != self.checksum:
            raise ChecksumValidationFailureError("Reconstruct checksum validation failed")
        doc_item.validate_checksum()
        return doc_item

    def to_metadata_packet(self) -> 'MetadataPacket':
        """
        Converts the DataPacket to a MetadataPacket.

        Creates a MetadataPacket instance containing the record_id and metadata,
        and verifies its integrity.

        Returns:
            MetadataPacket instance created from the DataPacket.

        Raises:
            ChecksumValidationFailureError: If the converted data fails checksum validation.
        """
        meta_item = MetadataPacket(record_id=self.record_id, metadata=self.metadata)
        self.validate_checksum()
        reconstructed = DataPacket(type=self.type, record_id=meta_item.record_id, original_data=self.original_data, vector=self.vector, metadata=meta_item.metadata)
        if reconstructed.checksum != self.checksum:
            raise ChecksumValidationFailureError("Reconstruct checksum validation failed")
        meta_item.validate_checksum()
        return meta_item

    def __eq__(self, other):
        """
        Compares two DataPacket instances for equality.

        Handles the case where shape might be represented as a list in one instance
        and a tuple in another (due to JSON serialization and deserialization).

        Args:
            other: Another object to compare with.

        Returns:
            bool: True if the instances are equal, False otherwise.
        """
        if not isinstance(other, type(self)):
            return False

        # Compare vectors using np.array_equal
        vectors_equal = ((self.vector is None and other.vector is None) or
                         (self.vector is not None and other.vector is not None and
                          np.array_equal(self.vector, other.vector)))

        # Check all other fields
        return (self.type == other.type and
                self.record_id == other.record_id and
                self.checksum_algorithm == other.checksum_algorithm and
                self.original_data == other.original_data and
                vectors_equal and
                self.metadata == other.metadata and
                self.checksum == other.checksum)