import base64
from typing import Dict, Any, List, Optional

import numpy as np
from pydantic import BaseModel, Field


class NumpyArray(BaseModel):
    """Model for serialized numpy array"""
    b64: str = Field(..., description="Base64 encoded array data")
    dtype: str = Field(..., description="NumPy dtype as string")
    shape: List[int] = Field(..., description="Array shape")

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        raw = base64.b64decode(self.b64.encode('ascii'))
        dtype = np.dtype(self.dtype)
        return np.frombuffer(raw, dtype=dtype).reshape(self.shape)

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'NumpyArray':
        """Create from numpy array"""
        return cls(
            b64=base64.b64encode(array.tobytes()).decode('ascii'),
            dtype=str(array.dtype),
            shape=list(array.shape)
        )


class DataPacketModel(BaseModel):
    """Model for DataPacket API representation"""
    type: str = Field(..., description="Type of data packet")
    record_id: str = Field(..., description="Unique record identifier")
    original_data: Optional[Any] = Field(None, description="Original data")
    vector: Optional[NumpyArray] = Field(None, description="Vector data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")
    checksum_algorithm: str = Field(default='sha256', description="Checksum algorithm")
    checksum: Optional[str] = Field(None, description="Customer-provided checksum for validation")


class QueryPacketModel(BaseModel):
    """Model for QueryPacket API representation"""
    query_vector: NumpyArray = Field(..., description="Query vector")
    k: int = Field(..., description="Number of results to return")
    where: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    where_document: Optional[Dict[str, Any]] = Field(None, description="Document filter")
    checksum_algorithm: str = Field(default='sha256', description="Checksum algorithm")
    checksum: Optional[str] = Field(None, description="Customer-provided checksum for validation")


class InsertRequest(BaseModel):
    """Request model for inserting data"""
    packet: DataPacketModel


class SearchRequest(BaseModel):
    """Request model for searching"""
    query: QueryPacketModel
