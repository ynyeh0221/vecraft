import logging
import os
import time

from vecraft_db.core.lock.mvcc_manager import CollectionVersion


class SnapshotManager:
    """Manages the persistence of collection indexes through snapshots.

    This class provides functionality to save and load snapshots of collection indexes
    (vector, metadata, and document indexes) to and from the disk. It supports atomic
    operations through temporary files to ensure data integrity during persistence.

    The manager handles three types of index snapshots:
    1. Vector index snapshots (.idxsnap)
    2. Metadata index snapshots (.metasnap)
    3. Document index snapshots (.docsnap)
    4. LSN (Log Sequence Number) metadata (.lsnmeta)

    Attributes:
        _collection_metadata: Dictionary containing metadata for all collections.
        _mvcc_manager: Manager for multi-version concurrency control.
        _logger: Logger instance for recording diagnostic information.
    """
    def __init__(self, collection_metadata, mvcc_manager, logger=None):
        self._collection_metadata = collection_metadata
        self._mvcc_manager = mvcc_manager
        self._logger = logger or logging.getLogger(__name__)

    def save_snapshots(self, name: str, version: CollectionVersion):
        """Save snapshots for a collection to disk atomically.

        This method saves the vector, metadata, and document indexes to disk using
        temporary files to ensure atomicity. It first writes to temporary files and
        then atomically replaces the main snapshot files.

        Args:
            name: Name of the collection to save snapshots for.
            version: CollectionVersion object containing the indexes to snapshot.

        Note:
            This operation is atomic - either all snapshots are updated or none.
        """
        self._logger.info(f"Saving snapshots for collection {name}")
        start_time = time.time()

        # write .tempsnap → main
        self.flush_indexes(name, version, to_temp_files=True)

        self._logger.info(f"Successfully saved all snapshots for collection {name} in {time.time() - start_time:.2f}s")

    def load_snapshots(self, name: str, version: CollectionVersion) -> bool:
        """Load snapshots for a collection from disk.

        Attempts to load vector, metadata, and document index snapshots for the
        given collection. All three snapshots must exist for loading to succeed.

        Args:
            name: Name of the collection to load snapshots for.
            version: CollectionVersion object where the loaded indexes will be stored.

        Returns:
            bool: True if all snapshots were successfully loaded, False if any snapshot
                  files are missing.

        Note:
            This method does not raise exceptions for missing snapshot files but
            returns False instead.
        """
        self._logger.info(f"Attempting to load snapshots for collection {name}")
        metadata = self._collection_metadata[name]
        vec_snap, meta_snap, doc_snap = metadata['vec_snap'], metadata['meta_snap'], metadata['doc_snap']

        if vec_snap.exists() and meta_snap.exists() and doc_snap.exists():
            start_time = time.time()

            # vector index
            vec_data = vec_snap.read_bytes()
            version.vec_index.deserialize(vec_data)
            self._logger.debug(f"Loaded vector snapshot ({len(vec_data)} bytes)")

            # metadata index
            meta_data = meta_snap.read_bytes()
            version.meta_index.deserialize(meta_data)
            self._logger.debug(f"Loaded metadata snapshot ({len(meta_data)} bytes)")

            # document index
            doc_data = doc_snap.read_bytes()
            version.doc_index.deserialize(doc_data)
            self._logger.debug(f"Loaded document snapshot ({len(doc_data)} bytes)")

            self._logger.info(f"Successfully loaded all snapshots for collection {name} in {time.time() - start_time:.2f}s")
            return True

        self._logger.info(f"Snapshots not found for collection {name}")
        return False

    def flush_indexes(self, name: str, version: CollectionVersion, to_temp_files: bool = True):
        """Flush in-memory indexes to snapshot files.

        Serializes the vector, metadata, and document indexes to disk, along with
        LSN metadata. Can write directly to the main snapshot files or use temporary
        files for atomic operations.

        Args:
            name: Name of the collection to flush indexes for.
            version: CollectionVersion object containing the indexes to flush.
            to_temp_files: If True, save to temporary files and then atomically replace
                          the main snapshots; otherwise write directly to the main files.
                          Defaults to True.

        Raises:
            Exception: Any exception that occurs during serialization or file operations.
                      If using temporary files, all created temporary files are cleaned up
                      on error.

        Note:
            When using temporary files (to_temp_files=True), this operation is atomic -
            either all snapshots are updated or none. Each file is also fsync'ed to ensure
            durability.
        """
        metadata = self._collection_metadata[name]

        tempfile_suffix = '.tempsnap'
        if to_temp_files:
            # preserve the original suffix and append ".tempsnap"
            vec_path = metadata['vec_snap'].with_suffix(metadata['vec_snap'].suffix + tempfile_suffix)
            meta_path = metadata['meta_snap'].with_suffix(metadata['meta_snap'].suffix + tempfile_suffix)
            doc_path = metadata['doc_snap'].with_suffix(metadata['doc_snap'].suffix + tempfile_suffix)
            lsn_path  = metadata['lsn_meta'].with_suffix(metadata['lsn_meta'].suffix + tempfile_suffix)
        else:
            vec_path = metadata['vec_snap']
            meta_path = metadata['meta_snap']
            doc_path = metadata['doc_snap']
            lsn_path  = metadata['lsn_meta']

        try:
            # serialize and write
            vec_data = version.vec_index.serialize()
            vec_path.write_bytes(vec_data)

            meta_data = version.meta_index.serialize()
            meta_path.write_bytes(meta_data)

            doc_data = version.doc_index.serialize()
            doc_path.write_bytes(doc_data)

            # Write visible_lsn to metadata file
            import json
            lsn_data = json.dumps({
                "visible_lsn": self._mvcc_manager.visible_lsn.get(name, 0),
                "timestamp": time.time()
            }).encode('utf-8')
            lsn_path.write_bytes(lsn_data)

            # fsync each
            for p in (vec_path, meta_path, doc_path, lsn_path):
                with open(p, 'rb') as f:
                    os.fsync(f.fileno())

            # atomic replace temps nap → main
            if to_temp_files:
                os.replace(vec_path, metadata['vec_snap'])
                os.replace(meta_path, metadata['meta_snap'])
                os.replace(doc_path, metadata['doc_snap'])
                os.replace(lsn_path,  metadata['lsn_meta'])

        except Exception:
            # clean up on error
            if to_temp_files:
                for temp_file in (vec_path, meta_path, doc_path, lsn_path):
                    if temp_file.exists():
                        temp_file.unlink()
            raise