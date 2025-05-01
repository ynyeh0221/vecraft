import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.vecraft.core.storage_interface import StorageEngine
from src.vecraft.core.vector_type_interface import VectorType
from src.vecraft.metadata.schema import CollectionSchema


class Collection:
    def __init__(self, name: str, schema: CollectionSchema, storage: StorageEngine, index_factory,
                 vector_type: VectorType):
        self.name = name
        self.schema = schema
        self._storage = storage
        self._index = index_factory(kind="brute_force", dim=schema.field.dim)
        self._vector_type = vector_type
        self._config_file = Path(f"{name}_config.json")
        self._config = self._load_config()

        # Optionally rebuild index from storage if needed
        self._rebuild_index()

    def insert(self, raw: Any, metadata: Dict[str, Any], record_id: int = None) -> int:
        """
        Insert or update a record in the collection.

        Args:
            raw: The raw vector data to encode
            metadata: User-provided metadata to associate with this vector
            record_id: Optional record ID for updates, generated if not provided

        Returns:
            The record ID
        """
        # Encode vector
        vec = self._vector_type.encode(raw)

        # Generate or use record ID
        if record_id is None:
            record_id = self.get_next_id()

        # Serialize data
        vector_bytes = vec.tobytes()
        metadata_bytes = json.dumps(metadata).encode('utf-8')

        # Create header
        header = np.array([record_id, len(vector_bytes), len(metadata_bytes)], dtype=np.int32).tobytes()

        # Combine into record
        record_bytes = header + vector_bytes + metadata_bytes
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
        self._index.add(vec, id_=record_id)

        return record_id

    def search(self, query_raw: Any, k: int) -> List[Dict[str, Any]]:
        """
        Search for similar vectors and return complete records.

        Args:
            query_raw: The raw query data to search for
            k: Number of results to return

        Returns:
            List of matching records with their data and similarity scores
        """
        # Encode the query vector
        qvec = self._vector_type.encode(query_raw)

        # Get raw search results (id, distance pairs)
        raw_results = self._index.search(qvec, k)

        # Convert to complete records
        results = []
        for record_id, distance in raw_results:
            # Get the complete record
            record = self.get(record_id)
            if record:
                # Add the distance score to the record
                record['distance'] = distance
                results.append(record)

        return results

    def get(self, record_id: int) -> dict:
        """Retrieve a record by ID."""
        record_location = self.get_record_location(record_id)
        if not record_location:
            return None

        # Read record data
        record_data = self._storage.read(record_location['offset'], record_location['size'])

        # Parse header
        header_size = 3 * 4  # 3 int32 values
        header = np.frombuffer(record_data[:header_size], dtype=np.int32)

        id_ = header[0]
        vector_size = header[1]
        metadata_size = header[2]

        # Extract vector and metadata
        vector_start = header_size
        vector_end = vector_start + vector_size
        vector_bytes = record_data[vector_start:vector_end]

        metadata_start = vector_end
        metadata_end = metadata_start + metadata_size
        metadata_bytes = record_data[metadata_start:metadata_end]

        # Decode
        vec = np.frombuffer(vector_bytes, dtype=np.float32)
        user_metadata = json.loads(metadata_bytes.decode('utf-8'))

        # Decode vector back to original format if possible
        original_data = None
        if hasattr(self._vector_type, 'decode'):
            try:
                original_data = self._vector_type.decode(vec)
            except Exception as e:
                # If decoding fails, just use the raw vector
                pass

        return {
            'id': id_,
            'vector': vec,
            'original_data': original_data,
            'user_metadata': user_metadata
        }

    def delete(self, record_id: int) -> bool:
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
            header_size = 3 * 4  # 3 int32 values
            header = np.frombuffer(record_data[:header_size], dtype=np.int32)

            vector_size = header[1]

            # Extract vector
            vector_bytes = record_data[header_size:header_size + vector_size]
            vec = np.frombuffer(vector_bytes, dtype=np.float32)

            # Add to index
            self._index.add(vec, id_=int(record_id))

    def get_next_id(self) -> int:
        """Get the next available ID for this collection."""
        next_id = self._config.get('next_id', 0)
        self._config['next_id'] = next_id + 1
        self._save_config()
        return next_id

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

    def get_record_location(self, record_id: int) -> dict:
        """Get storage location for a specific record."""
        record_locations = self._config.get('records', {})
        return record_locations.get(str(record_id))

    def add_record_location(self, record_id: int, offset: int, size: int) -> None:
        """Add record storage location."""
        if 'records' not in self._config:
            self._config['records'] = {}

        self._config['records'][str(record_id)] = {
            'offset': offset,
            'size': size
        }

        self._save_config()

    def update_record_location(self, record_id: int, offset: int, size: int) -> None:
        """Update the storage location for an existing record."""
        if 'records' not in self._config:
            self._config['records'] = {}

        self._config['records'][str(record_id)] = {
            'offset': offset,
            'size': size
        }

        self._save_config()

    def mark_location_deleted(self, record_id: int) -> None:
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

    def delete_record_location(self, record_id: int) -> None:
        """Remove record location information."""
        if 'records' in self._config and str(record_id) in self._config['records']:
            del self._config['records'][str(record_id)]
            self._save_config()

    def get_all_record_locations(self) -> dict:
        """Get all record storage locations."""
        return self._config.get('records', {})
