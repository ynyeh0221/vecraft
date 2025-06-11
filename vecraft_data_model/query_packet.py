"""Data structures for representing query requests."""


import base64
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import numpy as np

from vecraft_data_model.checksum_util import get_checksum_func, _prepare_field_bytes, _concat_bytes
from vecraft_exception_model.exception import ChecksumValidationFailureError


@dataclass
class QueryPacket:
    """A packet describing a query against a collection.

    Attributes:
        query_vector: Vector used for the search operation.
        k: Number of nearest neighbors to return.
        where: Optional filter on record metadata.
        where_document: Optional filter on document contents.
        checksum_algorithm: Algorithm used to compute ``checksum``.
        checksum: Hex encoded checksum for verifying integrity.
    """

    query_vector: np.ndarray
    k: int
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None
    checksum_algorithm: str = 'sha256'

    checksum: str = field(init=False)

    def __post_init__(self):
        """Compute and store the checksum after initialization."""
        func = get_checksum_func(self.checksum_algorithm)
        raw = self._serialize_for_checksum()
        self.checksum = func(raw)

    def _serialize_for_checksum(self) -> bytes:
        """Serialize fields into bytes for checksum calculation."""
        parts: List[bytes] = [
            self.query_vector.tobytes(),
            _prepare_field_bytes(self.k),
            _prepare_field_bytes(self.where or {}),
            _prepare_field_bytes(self.where_document or {})
        ]
        return _concat_bytes(parts)

    def validate_checksum(self) -> bool:
        """Verify that the checksum matches the serialized field bytes."""
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
