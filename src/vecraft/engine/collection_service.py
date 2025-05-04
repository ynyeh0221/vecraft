import json
import pickle
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable

import numpy as np

from src.vecraft.analysis.tsne import generate_tsne
from src.vecraft.core.data import DataPacket, QueryPacket
from src.vecraft.core.index_interface import IndexItem
from src.vecraft.engine.locks import ReentrantRWLock, write_locked_attr, read_locked_attr
from src.vecraft.index.record_metadata.metadata_index import MetadataItem
from src.vecraft.index.record_vector.document_filter_evaluator import DocumentFilterEvaluator
from src.vecraft.metadata.catalog import JsonCatalog
from src.vecraft.metadata.schema import CollectionSchema


class CollectionService:
    def __init__(
        self,
        catalog: JsonCatalog,
        wal_factory: Callable,
        storage_factory: Callable,
        vector_index_factory: Callable,
        metadata_index_factory: Callable
    ):
        self._rwlock = ReentrantRWLock()
        self._catalog = catalog
        self._wal_factory = wal_factory
        self._storage_factory = storage_factory
        self._vector_index_factory = vector_index_factory
        self._metadata_index_factory = metadata_index_factory

        # resources per collection_name
        self._collections: Dict[str, Dict[str, Any]] = {}

    def _init_collection(self, name: str):
        # create on-demand
        if name in self._collections:
            return
        schema: CollectionSchema = self._catalog.get_schema(name)
        wal = self._wal_factory(f"{name}.wal")
        vec_snap = Path(f"{name}.idxsnap")
        meta_snap = Path(f"{name}.metasnap")
        storage = self._storage_factory(data_path=f"{name}_storage.json", index_path=f"{name}_location_index.json")
        vector_index = self._vector_index_factory(kind="hnsw", dim=schema.field.dim)
        meta_index = self._metadata_index_factory()

        # load or rebuild
        if vec_snap.exists() and meta_snap.exists():
            vec_data = pickle.loads(vec_snap.read_bytes())
            vector_index.deserialize(vec_data)
            meta_data = meta_snap.read_bytes()
            meta_index.deserialize(meta_data)
        else:
            # full rebuild loops
            for rid, loc in storage.get_all_record_locations().items():
                data = storage.read(loc['offset'], loc['size'])
                _, orig_size, vec_size, _ = struct.unpack('<4I', data[:16])
                vec_start = 16 + orig_size
                vec = np.frombuffer(data[vec_start:vec_start+vec_size], dtype=np.float32)
                vector_index.add(IndexItem(record_id=str(rid), vector=vec))
            for rid in storage.get_all_record_locations().keys():
                rec_data = self.get(name, rid)
                if rec_data and 'metadata' in rec_data:
                    meta_index.add(MetadataItem(record_id=rid, metadata=rec_data['metadata']))

        # replay WAL
        self._rwlock.acquire_write()
        try:
            wal.replay(lambda entry: self._replay_entry(name, entry))
            wal.clear()
        finally:
            self._rwlock.release_write()

        # store resources
        self._collections[name] = {
            'schema': schema,
            'wal': wal,
            'storage': storage,
            'vec_index': vector_index,
            'meta_index': meta_index,
            'vec_snap': vec_snap,
            'meta_snap': meta_snap
        }

    def _load_snapshots(self, name: str) -> bool:
        """Load vector index and metadata snapshots for the given collection, if they exist."""
        res = self._collections[name]
        vec_snap = res['vec_snap']
        meta_snap = res['meta_snap']
        if vec_snap.exists() and meta_snap.exists():
            # vector index
            vec_data = pickle.loads(vec_snap.read_bytes())
            res['vec_index'].deserialize(vec_data)
            # metadata index
            meta_data = meta_snap.read_bytes()
            res['meta_index'].deserialize(meta_data)
            return True
        return False

    def _save_snapshots(self, name: str):
        res = self._collections[name]
        res['vec_snap'].write_bytes(res['vec_index'].serialize())
        res['meta_snap'].write_bytes(res['meta_index'].serialize())

    def _replay_entry(self, name: str, entry: dict) -> None:
        data_packet = DataPacket.from_dict(entry)
        data_packet.validate()
        if data_packet.type == "insert":
            self._apply_insert(name, data_packet)
        elif data_packet.type == "delete":
            self._apply_delete(name, data_packet)
        data_packet.validate()

    def _apply_insert(self, name: str, data_packet: DataPacket) -> None:
        res = self._collections[name]
        record_id = data_packet.record_id
        orig = data_packet.original_data
        vec = data_packet.vector
        meta = data_packet.metadata

        orig_b = json.dumps(orig).encode('utf-8')
        vec_b = vec.tobytes()
        meta_b = json.dumps(meta).encode('utf-8')
        header = struct.pack('<4I', len(record_id), len(orig_b), len(vec_b), len(meta_b))
        rec_bytes = header + orig_b + vec_b + meta_b
        size = len(rec_bytes)

        old_loc = res['storage'].get_record_location(record_id)
        if old_loc:
            old = self.get(name, record_id)
            old_vec = old.get('vector')
            old_meta = old.get('metadata', {})
        else:
            old_vec = old_meta = None

        wrote_storage = updated_loc = updated_meta = updated_vec = False

        try:
            # A) write storage
            all_locs = res['storage'].get_all_record_locations()
            new_offset = max((l['offset'] + l['size'] for l in all_locs.values()), default=0)
            actual_offset = res['storage'].write(rec_bytes, new_offset)
            wrote_storage = True

            # B) location index
            if old_loc:
                res['storage'].mark_deleted(record_id)
                res['storage'].delete_record(record_id)
            res['storage'].add_record(record_id, new_offset, size)
            updated_loc = True

            # C) metadata
            if old_meta is not None:
                res['meta_index'].update(
                    MetadataItem(record_id=record_id, metadata=old_meta),
                    MetadataItem(record_id=record_id, metadata=meta)
                )
            else:
                res['meta_index'].add(MetadataItem(record_id=record_id, metadata=meta))
            updated_meta = True

            # D) vector index
            res['vec_index'].add(IndexItem(record_id=record_id, vector=vec))
            updated_vec = True

        except Exception:
            # rollback
            if updated_vec:
                res['vec_index'].delete(record_id=record_id)
                if old_vec is not None:
                    res['vec_index'].add(IndexItem(record_id=record_id, vector=old_vec))

            if updated_meta:
                res['meta_index'].delete(MetadataItem(record_id=record_id, metadata=meta))
                if old_meta is not None:
                    res['meta_index'].add(MetadataItem(record_id=record_id, metadata=old_meta))

            if updated_loc:
                res['storage'].mark_deleted(record_id)
                res['storage'].delete_record(record_id)
                if old_loc is not None:
                    res['storage'].add_record(record_id, old_loc['offset'], old_loc['size'])

            raise

    def _apply_delete(self, name: str, data_packet: DataPacket) -> None:
        res = self._collections[name]
        record_id = data_packet.record_id
        old_loc = res['storage'].get_record_location(record_id)
        if not old_loc:
            return
        rec = self.get(name, record_id)
        old_vec = rec.get('vector')
        old_meta = rec.get('metadata', {})

        removed_location = removed_meta = removed_vec = False

        try:
            res['storage'].mark_deleted(record_id)
            removed_location = True

            res['storage'].delete_record(record_id)
            res['meta_index'].delete(MetadataItem(record_id=record_id, metadata=old_meta))
            removed_meta = True

            res['vec_index'].delete(record_id=record_id)
            removed_vec = True

        except Exception:
            if removed_vec:
                res['vec_index'].add(IndexItem(record_id=record_id, vector=old_vec))

            if removed_meta:
                res['meta_index'].add(MetadataItem(record_id=record_id, metadata=old_meta))

            if removed_location:
                res['storage'].add_record(record_id, old_loc['offset'], old_loc['size'])

            raise

    @write_locked_attr('_rwlock')
    def insert(self, collection: str, data_packet: DataPacket) -> str:
        self._init_collection(collection)

        print(f"Inserting {data_packet.record_id} to {collection} started")

        data_packet.validate()
        schema: CollectionSchema = self._collections[collection]['schema']
        if len(data_packet.vector) != schema.field.dim:
            raise ValueError(f"Vector dimension mismatch: expected {schema.field.dim}, got {len(data_packet.vector)}")
        self._collections[collection]['wal'].append(data_packet)
        self._apply_insert(collection, data_packet)
        data_packet.validate()

        print(f"Inserting {data_packet.record_id} to {collection} completed")

        return data_packet.record_id

    @write_locked_attr('_rwlock')
    def delete(self, collection: str, data_packet: DataPacket) -> bool:
        self._init_collection(collection)

        print(f"Deleting {data_packet.record_id} from {collection} started")

        data_packet.validate()
        self._collections[collection]['wal'].append(data_packet)
        self._apply_delete(collection, data_packet)
        data_packet.validate()

        print(f"Deleting {data_packet.record_id} from {collection} completed")

        return True

    @read_locked_attr('_rwlock')
    def search(self, collection: str, query_packet: QueryPacket) -> List[Dict[str, Any]]:
        self._init_collection(collection)

        print(f"Searching from {collection} started")

        schema: CollectionSchema = self._collections[collection]['schema']
        if len(query_packet.query_vector) != schema.field.dim:
            raise ValueError(
                f"Query dimension mismatch: expected {schema.field.dim}, got {len(query_packet.query_vector)}"
            )
        allowed_ids: Optional[Set[str]] = None

        # 1) MetadataIndex
        if query_packet.where:
            metadata_ids = self._collections[collection]['meta_index'].get_matching_ids(query_packet.where)
            if metadata_ids is not None:
                allowed_ids = metadata_ids
                if not allowed_ids:
                    return []

        if query_packet.where_document:
            doc_ids = self._filter_by_document(collection, query_packet.where_document, allowed_ids)
            allowed_ids = allowed_ids & doc_ids if allowed_ids is not None else doc_ids
            if not allowed_ids:
                return []

        # 2) VectorIndex
        raw_results = self._collections[collection]['vec_index'].search(query_packet.query_vector, query_packet.k, allowed_ids=allowed_ids)

        # 3) LocationIndex + Storage inside get()
        results: List[Dict[str, Any]] = []
        for rec_id, dist in raw_results:
            rec = self.get(collection, rec_id)
            if not rec:
                continue
            rec['distance'] = dist
            results.append(rec)

        query_packet.validate()

        print(f"Searching from {collection} completed")

        return results

    @read_locked_attr('_rwlock')
    def get(self, collection: str, record_id: str) -> dict:
        self._init_collection(collection)

        print(f"Getting {record_id} from {collection} started")

        loc = self._collections[collection]['storage'].get_record_location(record_id)
        if not loc:
            return {}

        data = self._collections[collection]['storage'].read(loc['offset'], loc['size'])
        header_size = 4 * 4
        rid_len, original_data_size, vector_size, metadata_size = struct.unpack(
            '<4I',
            data[:header_size]
        )
        pos = header_size
        orig_bytes = data[pos:pos+original_data_size]
        pos += original_data_size
        vec_bytes = data[pos:pos+vector_size]
        pos += vector_size
        meta_bytes = data[pos:pos+metadata_size]

        print(f"Getting {record_id} from {collection} completed")

        return {
            'id': record_id,
            'original_data': json.loads(orig_bytes.decode('utf-8')),
            'vector': np.frombuffer(vec_bytes, dtype=np.float32),
            'metadata': json.loads(meta_bytes.decode('utf-8'))
        }

    @write_locked_attr('_rwlock')
    def flush(self):
        for name in list(self._collections.keys()):
            print(f"Flushing {name} started")
            self._collections[name]['storage'].flush()
            self._save_snapshots(name)
            print(f"Flushing {name} completed")

    @write_locked_attr('_rwlock')
    def _rebuild_index(self, name: str) -> None:
        """Rebuild the vector index for the given collection from storage."""
        res = self._collections[name]
        for rid, loc in res['storage'].get_all_record_locations().items():
            data = res['storage'].read(loc['offset'], loc['size'])
            # unpack header
            _, orig_size, vec_size, _ = struct.unpack('<4I', data[:16])
            vec_start = 16 + orig_size
            vec = np.frombuffer(data[vec_start:vec_start + vec_size], dtype=np.float32)
            res['index'].add(IndexItem(record_id=str(rid), vector=vec))

    @write_locked_attr('_rwlock')
    def _rebuild_metadata_index(self, name: str) -> None:
        """Rebuild the metadata index for the given collection from storage."""
        res = self._collections[name]
        for rid in res['storage'].get_all_record_locations().keys():
            rec = self.get(name, rid)
            if rec and 'metadata' in rec:
                res['meta_index'].add(MetadataItem(record_id=rid, metadata=rec['metadata']))

    @read_locked_attr('_rwlock')
    def _filter_by_document(self,
                            name: str,
                            filter_condition: Dict[str, Any],
                            allowed_ids: Optional[Set[str]] = None) -> Set[str]:
        evaluator = DocumentFilterEvaluator()
        matching = set()
        candidates = allowed_ids if allowed_ids else set(self._collections[name]['vec_index'].get_all_ids())
        for rid in candidates:
            rec = self.get(name, rid)
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

    @read_locked_attr('_rwlock')
    def generate_tsne_plot(
            self,
            name: str,
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
            record_ids = list(self._collections[name]['storage'].get_all_record_locations().keys())

        vectors = []
        labels = []
        for rid in record_ids:
            rec = self.get(name, rid)
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
