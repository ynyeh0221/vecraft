import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.vecraft.core.index_item import IndexItem
from src.vecraft.core.storage_interface import StorageEngine
from src.vecraft.index.document_filter_evaluator import DocumentFilterEvaluator
from src.vecraft.metadata.schema import CollectionSchema


class Collection:
    def __init__(self, name: str, schema: CollectionSchema, storage: StorageEngine, index_factory):
        self.name = name
        self.schema = schema
        self._storage = storage
        self._index = index_factory(kind="brute_force", dim=schema.field.dim)
        self._config_file = Path(f"{name}_config.json")
        self._config = self._load_config()

        # Optionally rebuild index from storage if needed
        self._rebuild_index()

    def insert(self, original_data: Any, vector: np.ndarray, metadata: Dict[str, Any], record_id: str = None) -> str:
        """
        Insert or update a record in the collection.

        Args:
            original_data: The original data to store
            vector: The pre-encoded vector representing the data
            metadata: User-provided metadata to associate with this vector
            record_id: Optional record ID for updates, generated if not provided

        Returns:
            The record ID
        """
        # Validate vector dimensions
        if len(vector) != self.schema.field.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.schema.field.dim}, got {len(vector)}")

        # Generate or use record ID
        if record_id is None:
            record_id = self.get_next_id()

        # Serialize data
        original_data_bytes = json.dumps(original_data).encode('utf-8')
        vector_bytes = vector.tobytes()
        metadata_bytes = json.dumps(metadata).encode('utf-8')

        # Create header
        header = np.array([record_id, len(original_data_bytes), len(vector_bytes), len(metadata_bytes)],
                          dtype=np.int32).tobytes()

        # Combine into record
        record_bytes = header + original_data_bytes + vector_bytes + metadata_bytes
        record_size = len(record_bytes)

        # Check if this is an update
        record_location = self.get_record_location(record_id)

        if record_location:
            # This is an update
            if record_size <= record_location['size']:
                # Overwrite in place
                self._storage.write(record_bytes, record_location['offset'])
                self.update_record_location(record_id, record_location['offset'], record_size)
            else:
                # Mark old space as deleted
                self.mark_location_deleted(record_id)

                # Write at end
                offset = self.get_next_offset()
                self._storage.write(record_bytes, offset)
                self.update_record_location(record_id, offset, record_size)
        else:
            # New insert
            offset = self.get_next_offset()
            self._storage.write(record_bytes, offset)
            self.add_record_location(record_id, offset, record_size)

        # Update index
        self._index.add(IndexItem(id=record_id, vector=vector))

        return record_id

    def search(self, query_vector: np.ndarray, k: int,
               where: Dict[str, Any] = None,
               where_document: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with filtering on metadata and/or document content.

        Args:
            query_vector: The pre-encoded query vector
            k: Number of results to return
            where: Optional dictionary specifying metadata filter conditions
            where_document: Optional dictionary specifying document content filter conditions

        Returns:
            List of matching records with their data and similarity scores
        """
        # Validate vector dimensions
        if len(query_vector) != self.schema.field.dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.schema.field.dim}, got {len(query_vector)}")

        # Pre-filter phase
        allowed_ids = None

        # If we have metadata filter
        """
        if where:
            # Try to get matching IDs using metadata index
            metadata_matching_ids = self._metadata_index.get_matching_ids(where)

            # If we couldn't use the metadata index effectively, filter manually
            if metadata_matching_ids is None:
                metadata_matching_ids = self._filter_by_metadata(where)

            # Initialize allowed_ids with metadata matches
            allowed_ids = metadata_matching_ids
        """

        # If we have document content filter
        if where_document and (allowed_ids is None or allowed_ids):
            # Get IDs that match document filter
            document_matching_ids = self._filter_by_document(where_document, allowed_ids)

            # Update allowed_ids
            if allowed_ids is None:
                allowed_ids = document_matching_ids
            else:
                # Intersection of metadata and document filters
                allowed_ids = allowed_ids.intersection(document_matching_ids)

        # If no matches after filtering, return empty results
        if allowed_ids is not None and not allowed_ids:
            return []

        # Perform search with pre-filtering
        raw_results = self._index.search(query_vector, k, allowed_ids=allowed_ids)

        # Format and return results
        results = []
        for record_id, distance in raw_results:
            record = self.get(record_id)
            if record:
                record['distance'] = distance
                results.append(record)

        return results

    def _filter_by_document(self, filter_condition: Dict[str, Any],
                            allowed_ids: Optional[Set[str]] = None) -> Set[str]:
        """
        Filter records based on document content.

        Args:
            filter_condition: Document content filter condition
            allowed_ids: Optional set of IDs to filter within (for combined filtering)

        Returns:
            Set of matching record IDs
        """
        document_evaluator = DocumentFilterEvaluator()
        matching_ids = set()

        # Get candidate IDs
        candidate_ids = allowed_ids if allowed_ids else set(self._index.get_all_ids())

        # Check each document
        for record_id in candidate_ids:
            record = self.get(record_id)
            if record and 'original_data' in record:
                # Get document content
                document_content = record['original_data']

                # If document content is not a string, try to convert it
                if not isinstance(document_content, str):
                    if isinstance(document_content, dict):
                        # For dictionaries, convert to JSON string
                        try:
                            document_content = json.dumps(document_content)
                        except:
                            continue
                    else:
                        # Try string conversion for other types
                        try:
                            document_content = str(document_content)
                        except:
                            continue

                # Check if document matches filter
                if document_evaluator.matches(document_content, filter_condition):
                    matching_ids.add(record_id)

        return matching_ids

    def get(self, record_id: str) -> dict:
        """Retrieve a record by ID."""
        record_location = self.get_record_location(record_id)
        if not record_location:
            return None

        # Read record data
        record_data = self._storage.read(record_location['offset'], record_location['size'])

        # Parse header
        header_size = 4 * 4  # 4 int32 values
        header = np.frombuffer(record_data[:header_size], dtype=np.int32)

        id_ = header[0]
        original_data_size = header[1]
        vector_size = header[2]
        metadata_size = header[3]

        # Extract components
        current_pos = header_size

        # Extract original data
        original_data_bytes = record_data[current_pos:current_pos + original_data_size]
        current_pos += original_data_size

        # Extract vector
        vector_bytes = record_data[current_pos:current_pos + vector_size]
        current_pos += vector_size

        # Extract metadata
        metadata_bytes = record_data[current_pos:current_pos + metadata_size]

        # Decode
        original_data = json.loads(original_data_bytes.decode('utf-8'))
        vec = np.frombuffer(vector_bytes, dtype=np.float32)
        user_metadata = json.loads(metadata_bytes.decode('utf-8'))

        return {
            'id': id_,
            'original_data': original_data,
            'vector': vec,
            'user_metadata': user_metadata
        }

    def delete(self, record_id: str) -> bool:
        """Delete a record by ID."""
        record_location = self.get_record_location(record_id)
        if not record_location:
            return False

        # Mark space as deleted
        self.mark_location_deleted(record_id)

        # Remove from metadata
        self.delete_record_location(record_id)

        # Remove from index
        self._index.delete(id_=record_id)

        return True

    def flush(self):
        """Ensure all data and configuration are persisted."""
        self._save_config()
        self._storage.flush()

    def _load_config(self) -> dict:
        """Load collection configuration from disk."""
        if self._config_file.exists():
            return json.loads(self._config_file.read_text())
        else:
            # Initialize with default values
            config = {
                'next_id': 0,
                'records': {},
                'deleted_records': []
            }
            self._save_config(config)
            return config

    def _save_config(self, config=None):
        """Save collection configuration to disk."""
        if config is None:
            config = self._config
        self._config_file.write_text(json.dumps(config, indent=2))

    def _rebuild_index(self):
        """Rebuild the index from stored records."""
        records = self.get_all_record_locations()

        for record_id, record_location in records.items():
            # Read record data
            record_data = self._storage.read(record_location['offset'], record_location['size'])

            # Parse header
            header_size = 4 * 4  # 4 int32 values
            header = np.frombuffer(record_data[:header_size], dtype=np.int32)

            original_data_size = header[1]
            vector_size = header[2]

            # Calculate vector position
            vector_start = header_size + original_data_size

            # Extract vector
            vector_bytes = record_data[vector_start:vector_start + vector_size]
            vec = np.frombuffer(vector_bytes, dtype=np.float32)

            # Add to index
            self._index.add(vec, id_=int(record_id))

    def get_next_id(self) -> str:
        """Get the next available ID for this collection."""
        next_id = self._config.get('next_id', 0)
        self._config['next_id'] = next_id + 1
        self._save_config()
        return str(next_id)

    def get_next_offset(self) -> int:
        """Calculate the next available storage offset."""
        record_locations = self._config.get('records', {})

        if not record_locations:
            return 0

        # Find the end of the last record
        last_offset = 0
        last_size = 0

        for location_info in record_locations.values():
            record_offset = location_info['offset']
            record_size = location_info['size']
            if record_offset + record_size > last_offset + last_size:
                last_offset = record_offset
                last_size = record_size

        return last_offset + last_size

    def get_record_location(self, record_id: str) -> dict:
        """Get storage location for a specific record."""
        record_locations = self._config.get('records', {})
        return record_locations.get(str(record_id))

    def add_record_location(self, record_id: str, offset: int, size: int) -> None:
        """Add record storage location."""
        if 'records' not in self._config:
            self._config['records'] = {}

        self._config['records'][str(record_id)] = {
            'offset': offset,
            'size': size
        }

        self._save_config()

    def update_record_location(self, record_id: str, offset: int, size: int) -> None:
        """Update the storage location for an existing record."""
        if 'records' not in self._config:
            self._config['records'] = {}

        self._config['records'][str(record_id)] = {
            'offset': offset,
            'size': size
        }

        self._save_config()

    def mark_location_deleted(self, record_id: str) -> None:
        """Mark a record's storage location as deleted for potential space reuse."""
        if 'deleted_records' not in self._config:
            self._config['deleted_records'] = []

        record_location = self.get_record_location(record_id)
        if record_location:
            self._config['deleted_records'].append({
                'offset': record_location['offset'],
                'size': record_location['size']
            })

        self._save_config()

    def delete_record_location(self, record_id: str) -> None:
        """Remove record location information."""
        if 'records' in self._config and str(record_id) in self._config['records']:
            del self._config['records'][str(record_id)]
            self._save_config()

    def get_all_record_locations(self) -> dict:
        """Get all record storage locations."""
        return self._config.get('records', {})
