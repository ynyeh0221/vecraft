import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable

import numpy as np

from src.vecraft.analysis.tsne import generate_tsne
from src.vecraft.core.index_interface import IndexItem
from src.vecraft.core.storage_interface import StorageEngine
from src.vecraft.index.record_location.location_index_interface import RecordLocationIndex
from src.vecraft.engine.locks import RWLock
from src.vecraft.engine.transaction import Txn
from src.vecraft.index.document_filter_evaluator import DocumentFilterEvaluator
from src.vecraft.index.record_metadata.metadata_index import MetadataIndex, MetadataItem
from src.vecraft.metadata.schema import CollectionSchema
from src.vecraft.wal.wal_manager import WALManager


class Collection:
    def __init__(
        self,
        name: str,
        schema: CollectionSchema,
        storage: StorageEngine,
        index_factory: Callable,
        location_index: RecordLocationIndex
    ):
        self.name = name
        self.schema = schema
        self._storage = storage
        self._index = index_factory(kind="hnsw", dim=schema.field.dim)
        self._metadata_index = MetadataIndex()

        # per-record_location lock and txn
        self._lock = RWLock()
        self._txn = Txn(self._lock)

        # external record location record_vector
        self._location_index = location_index

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
        self._wal.clear()

    def _load_snapshots(self) -> bool:
        if self._vec_snap.exists() and self._meta_snap.exists():
            vec_data = pickle.loads(self._vec_snap.read_bytes())
            self._index.deserialize(vec_data)
            meta_data = self._meta_snap.read_bytes()
            self._metadata_index.deserialize(meta_data)
            return True
        return False

    def _save_snapshots(self) -> None:
        self._vec_snap.write_bytes(self._index.serialize())
        self._meta_snap.write_bytes(self._metadata_index.serialize())

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

        # stash old location
        old_loc = self._location_index.get_record_location(rid)
        wrote_storage = updated_config = added_vec_index = added_meta_index = False

        try:
            # append to storage
            records = self._location_index.get_all_record_locations()
            if records:
                new_offset = max(loc['offset'] + loc['size'] for loc in records.values())
            else:
                new_offset = 0
            self._storage.write(record_bytes, new_offset)
            wrote_storage = True

            # update config: tombstone old, add new
            if old_loc:
                self._location_index.mark_deleted(rid)
                self._location_index.delete_record(rid)

            self._location_index.add_record(rid, new_offset, size)
            updated_config = True

            # update indexes
            self._index.add(IndexItem(record_id=rid, vector=vec))
            added_vec_index = True
            self._metadata_index.add(MetadataItem(record_id=rid, metadata=meta))
            added_meta_index = True

        except Exception:
            # rollback
            if added_meta_index:
                self._metadata_index.delete(MetadataItem(record_id=rid, metadata=meta))
            if added_vec_index:
                self._index.delete(id_=rid)
            if updated_config:
                self._location_index.delete_record(rid)
                if old_loc:
                    self._location_index.add_record(rid, old_loc['offset'], old_loc['size'])
            if wrote_storage:
                self._location_index.mark_deleted(rid)
            raise

    def _apply_delete(self, record_id: str) -> None:
        old_loc = self._location_index.get_record_location(record_id)
        if not old_loc:
            return

        rec = self.get(record_id)
        old_vec = rec.get('vector')
        old_meta = rec.get('metadata', {})

        removed_location = removed_vec_index = removed_meta_index = False

        try:
            # mark tombstone
            self._location_index.mark_deleted(record_id)
            removed_location = True

            # delete pointer
            self._location_index.delete_record(record_id)

            # remove from in-memory indexes
            self._index.delete(id_=record_id)
            removed_vec_index = True
            self._metadata_index.delete(MetadataItem(record_id=record_id, metadata=old_meta))
            removed_meta_index = True

        except Exception:
            # rollback
            if removed_meta_index:
                self._metadata_index.add(MetadataItem(record_id=record_id, metadata=old_meta))
            if removed_vec_index:
                self._index.add(IndexItem(record_id=record_id, vector=old_vec))
            if removed_location:
                # restore record pointer (tombstone remains)
                self._location_index.add_record(record_id, old_loc['offset'], old_loc['size'])
            raise

    def insert(
        self,
        original_data: Any,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        record_id: str = None
    ) -> str:
        if len(vector) != self.schema.field.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.schema.field.dim}, got {len(vector)}")
        if record_id is None:
            record_id = self._location_index.get_next_id()
        entry = {
            "type": "insert",
            "record_id": record_id,
            "original_data": original_data,
            "vector": vector.tolist(),
            "metadata": metadata
        }
        self._wal.append(entry)
        self._apply_insert(entry)
        return record_id

    def delete(self, record_id: str) -> bool:
        entry = {"type": "delete", "record_id": record_id}
        self._wal.append(entry)
        self._apply_delete(record_id)
        return True

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        if len(query_vector) != self.schema.field.dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.schema.field.dim}, got {len(query_vector)}"
            )
        allowed_ids: Optional[Set[str]] = None

        if where:
            metadata_ids = self._metadata_index.get_matching_ids(where)
            if metadata_ids is not None:
                allowed_ids = metadata_ids
                if not allowed_ids:
                    return []

        if where_document:
            doc_ids = self._filter_by_document(where_document, allowed_ids)
            allowed_ids = allowed_ids & doc_ids if allowed_ids is not None else doc_ids
            if not allowed_ids:
                return []

        raw_results = self._index.search(query_vector, k, allowed_ids=allowed_ids)
        results: List[Dict[str, Any]] = []
        for rec_id, dist in raw_results:
            rec = self.get(rec_id)
            if not rec:
                continue
            rec['distance'] = dist
            results.append(rec)
        return results

    def get(self, record_id: str) -> dict:
        loc = self._location_index.get_record_location(record_id)
        if not loc:
            return {}
        data = self._storage.read(loc['offset'], loc['size'])
        header_size = 4 * 4
        header = np.frombuffer(data[:header_size], dtype=np.int32)
        original_data_size, vector_size, metadata_size = header[1], header[2], header[3]
        pos = header_size
        orig_bytes = data[pos:pos+original_data_size]; pos += original_data_size
        vec_bytes = data[pos:pos+vector_size]; pos += vector_size
        meta_bytes = data[pos:pos+metadata_size]
        return {
            'id': int(header[0]),
            'original_data': json.loads(orig_bytes.decode('utf-8')),
            'vector': np.frombuffer(vec_bytes, dtype=np.float32),
            'metadata': json.loads(meta_bytes.decode('utf-8'))
        }

    def flush(self):
        self._storage.flush()
        self._save_snapshots()

    def _rebuild_index(self):
        for rid, loc in self._location_index.get_all_record_locations().items():
            data = self._storage.read(loc['offset'], loc['size'])
            header = np.frombuffer(data[:16], dtype=np.int32)
            orig_size, vec_size = header[1], header[2]
            vec_start = 16 + orig_size
            vec = np.frombuffer(data[vec_start:vec_start+vec_size], dtype=np.float32)
            self._index.add(IndexItem(record_id=str(rid), vector=vec))

    def _rebuild_metadata_index(self):
        for rid in self._location_index.get_all_record_locations().keys():
            rec = self.get(rid)
            if rec and 'metadata' in rec:
                self._metadata_index.add(MetadataItem(record_id=rid, metadata=rec['metadata']))

    def _filter_by_document(self,
                            filter_condition: Dict[str, Any],
                            allowed_ids: Optional[Set[str]] = None) -> Set[str]:
        evaluator = DocumentFilterEvaluator()
        matching = set()
        candidates = allowed_ids if allowed_ids else set(self._index.get_all_ids())
        for rid in candidates:
            rec = self.get(rid)
            content = rec.get('original_data')
            if content is None:
                continue
            if not isinstance(content, str):
                try:
                    content = json.dumps(content)
                except:
                    content = str(content)
            if evaluator.matches(content, filter_condition):
                matching.add(rid)
        return matching

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
            record_ids = list(self._location_index.get_all_record_locations().keys())

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
