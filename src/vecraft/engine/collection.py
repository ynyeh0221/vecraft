import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable

import numpy as np

from src.vecraft.analysis.tsne import generate_tsne
from src.vecraft.core.index_interface import IndexItem
from src.vecraft.core.storage_interface import StorageEngine
from src.vecraft.engine.locks import RWLock
from src.vecraft.engine.transaction import Txn
from src.vecraft.index.document_filter_evaluator import DocumentFilterEvaluator
from src.vecraft.index.metadata_index import MetadataIndex, MetadataItem
from src.vecraft.metadata.schema import CollectionSchema
from src.vecraft.wal.wal_manager import WALManager


class Collection:
    def __init__(
        self,
        name: str,
        schema: CollectionSchema,
        storage: StorageEngine,
        index_factory: Callable
    ):
        self.name = name
        self.schema = schema
        self._storage = storage
        self._index = index_factory(kind="brute_force", dim=schema.field.dim)
        self._metadata_index = MetadataIndex()

        # per-collection lock and txn
        self._lock = RWLock()
        self._txn = Txn(self._lock)

        self._config_file = Path(f"{name}_config.json")
        self._config = self._load_config()

        # WAL and snapshots
        self._wal = WALManager(Path(f"{name}.wal"))
        self._vec_snap = Path(f"{name}.idxsnap")
        self._meta_snap = Path(f"{name}.metasnap")

        # Load snapshots if existed, else full rebuild
        if not self._load_snapshots():
            self._rebuild_index()
            self._rebuild_metadata_index()

        # Apply WAL entries since snapshot
        self._wal.replay(self._replay_entry)
        # Clear WAL after replay
        self._wal.clear()

    def _load_snapshots(self) -> bool:
        if self._vec_snap.exists() and self._meta_snap.exists():
            # load vector index
            vec_data = pickle.loads(self._vec_snap.read_bytes())
            self._index.deserialize(vec_data)
            # load metadata index
            meta_data = self._meta_snap.read_bytes()
            self._metadata_index.deserialize(meta_data)
            return True
        return False

    def _save_snapshots(self) -> None:
        # serialize and save vector index
        vec_bytes = self._index.serialize()
        self._vec_snap.write_bytes(vec_bytes)
        # serialize and save metadata index
        meta_bytes = self._metadata_index.serialize()
        self._meta_snap.write_bytes(meta_bytes)

    def _replay_entry(self, entry: dict) -> None:
        typ = entry.get("type")
        if typ == "insert":
            self._apply_insert(entry)
        elif typ == "delete":
            self._apply_delete(entry["record_id"])

    def _apply_insert(self, entry: dict) -> None:
        rid = entry["record_id"]
        orig = entry["original_data"]
        vec = np.array(entry["vector"], dtype=np.float32)
        meta = entry["metadata"]

        # Serialize into one bytes blob
        orig_bytes = json.dumps(orig).encode('utf-8')
        vec_bytes = vec.tobytes()
        meta_bytes = json.dumps(meta).encode('utf-8')
        header = np.array([
            int(rid),
            len(orig_bytes),
            len(vec_bytes),
            len(meta_bytes)
        ], dtype=np.int32).tobytes()
        record_bytes = header + orig_bytes + vec_bytes + meta_bytes
        size = len(record_bytes)

        # 1) Stash old location so we can restore on rollback
        old_loc = self.get_record_location(rid)

        wrote_storage = False
        updated_config = False
        added_vec_index = False
        added_meta_index = False

        try:
            # 2) Always append at EOF
            new_offset = self.get_next_offset()
            self._storage.write(record_bytes, new_offset)
            wrote_storage = True

            # 3) Update config: mark old as deleted, remove its pointer, then add the new one
            if old_loc:
                self.mark_location_deleted(rid)
                self.delete_record_location(rid)

            self.add_record_location(rid, new_offset, size)
            updated_config = True

            # 4) Update in-memory indexes
            self._index.add(IndexItem(record_id=rid, vector=vec))
            added_vec_index = True

            self._metadata_index.add(MetadataItem(record_id=rid, metadata=meta))
            added_meta_index = True

        except Exception:
            # Rollback in reverse order:

            if added_meta_index:
                self._metadata_index.delete(MetadataItem(record_id=rid, metadata=meta))

            if added_vec_index:
                self._index.delete(id_=rid)

            if updated_config:
                # remove the new pointer
                self.delete_record_location(rid)
                # undelete & restore the old pointer if it existed
                if old_loc:
                    self._config['deleted_records'].remove({
                        'offset': old_loc['offset'],
                        'size': old_loc['size']
                    })
                    self.add_record_location(rid, old_loc['offset'], old_loc['size'])

            if wrote_storage:
                # mark the newly-appended block deleted
                self.mark_location_deleted(rid)

            raise

    def _apply_delete(self, record_id: str) -> None:
        # 1) Retrieve and cache the old location and data
        old_loc = self.get_record_location(record_id)
        if not old_loc:
            return

        # Load the existing record so we can restore indexes if needed
        rec = self.get(record_id)
        old_vec = rec.get('vector')
        old_meta = rec.get('metadata', {})

        # Flags to track which steps have completed
        marked_deleted = False
        removed_location = False
        removed_vec_index = False
        removed_meta_index = False

        try:
            # 2) Mark the storage location as deleted
            self.mark_location_deleted(record_id)
            marked_deleted = True

            # 3) Remove the record pointer from config
            self.delete_record_location(record_id)
            removed_location = True

            # 4) Remove from the vector index
            self._index.delete(id_=record_id)
            removed_vec_index = True

            # 5) Remove from the metadata index
            self._metadata_index.delete(MetadataItem(record_id=record_id, metadata=old_meta))
            removed_meta_index = True

        except Exception:
            # Roll back in reverse order of execution

            if removed_meta_index:
                # 5) Restore metadata index
                self._metadata_index.add(MetadataItem(record_id=record_id, metadata=old_meta))

            if removed_vec_index:
                # 4) Restore vector index
                self._index.add(IndexItem(record_id=record_id, vector=old_vec))

            if removed_location:
                # 3) Restore the config pointer
                self.add_record_location(record_id, old_loc['offset'], old_loc['size'])

            if marked_deleted:
                # 2) Remove the tombstone so the old block is live again
                tombstone = {'offset': old_loc['offset'], 'size': old_loc['size']}
                deleted_records = self._config.get('deleted_records', [])
                if tombstone in deleted_records:
                    deleted_records.remove(tombstone)

            # Propagate the error so the caller knows to delete failed
            raise

    def insert(
        self,
        original_data: Any,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        record_id: str = None
    ) -> str:
        # Validate dimensions
        if len(vector) != self.schema.field.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.schema.field.dim}, got {len(vector)}")
        # Generate ID
        if record_id is None:
            record_id = self.get_next_id()
        # Prepare WAL entry
        entry = {
            "type": "insert",
            "record_id": record_id,
            "original_data": original_data,
            "vector": vector.tolist(),
            "metadata": metadata
        }
        # Write-ahead log
        self._wal.append(entry)
        # Apply insert
        self._apply_insert(entry)
        return record_id

    def delete(self, record_id: str) -> bool:
        # Prepare WAL
        entry = {"type": "delete", "record_id": record_id}
        self._wal.append(entry)
        # Apply delete
        self._apply_delete(record_id)
        return True

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with optional metadata and document filters.
        """
        # 1) Validate query vector dimension
        if len(query_vector) != self.schema.field.dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.schema.field.dim}, got {len(query_vector)}"
            )

        allowed_ids: Optional[Set[str]] = None

        # 2) Metadata filtering
        if where:
            metadata_ids = self._metadata_index.get_matching_ids(where)
            # If metadata_ids is not None, enforce filtering
            if metadata_ids is not None:
                allowed_ids = metadata_ids
                if not allowed_ids:
                    return []

        # 3) Document content filtering
        if where_document:
            doc_ids = self._filter_by_document(where_document, allowed_ids)
            if allowed_ids is None:
                allowed_ids = doc_ids
            else:
                allowed_ids &= doc_ids
            if not allowed_ids:
                return []

        # 4) Vector similarity search
        raw_results = self._index.search(query_vector, k, allowed_ids=allowed_ids)

        # 5) Fetch and format records
        results: List[Dict[str, Any]] = []
        for rec_id, dist in raw_results:
            rec = self.get(rec_id)
            if not rec:
                continue
            rec['distance'] = dist
            results.append(rec)
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
            return {}

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
        metadata = json.loads(metadata_bytes.decode('utf-8'))

        return {
            'id': id_,
            'original_data': original_data,
            'vector': vec,
            'metadata': metadata
        }

    def flush(self):
        """Ensure all data and configuration are persisted."""
        self._save_config()
        self._storage.flush()
        self._save_snapshots()

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
            self._index.add(IndexItem(record_id=str(record_id), vector=vec))

    def _rebuild_metadata_index(self):
        for rec_id_str, loc in self.get_all_record_locations().items():
            rec = self.get(rec_id_str)
            if rec and 'metadata' in rec:
                self._metadata_index.add(MetadataItem(record_id=rec_id_str, metadata=rec['metadata']))

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

    def read_transaction(self):
        """Publicly expose a read‐lock context."""
        return self._txn.read()

    def write_transaction(self):
        """Publicly expose a write‐lock context."""
        return self._txn.write()

    def generate_tsne_plot(
            self,
            record_ids: Optional[List[str]] = None,
            perplexity: int = 30,
            random_state: int = 42,
            outfile: str = "tsne.png"
    ) -> str:
        """
        Generate a t-SNE scatter plot for the given record IDs (or all records if None).

        Args:
            record_ids: Optional list of record IDs to visualize.
            perplexity: t-SNE perplexity parameter.
            random_state: Random seed for reproducibility.
            outfile: Path to save the generated PNG image.

        Returns:
            Path to the saved t-SNE plot image.
        """
        # Determine which IDs to plot
        if record_ids is None:
            record_ids = list(self.get_all_record_locations().keys())

        vectors = []
        labels = []
        for rid in record_ids:
            rec = self.get(rid)
            if not rec:
                continue
            vectors.append(rec['vector'])
            labels.append(rid)

        if not vectors:
            raise ValueError("No vectors available for t-SNE visualization.")

        # Stack into a 2D array
        data = np.vstack(vectors)

        # Call the helper to generate and save the plot
        return generate_tsne(
            vectors=data,
            labels=labels,
            outfile=outfile,
            perplexity=perplexity,
            random_state=random_state
        )
