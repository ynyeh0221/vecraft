from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.index_packets import LocationPacket
from vecraft_exception_model.exception import StorageFailureException


class WriteManager:
    """Manages write operations (inserts and deletes) to both storage and indexes.

    This class provides methods to handle the storage and indexing aspects of write
    operations separately, with built-in rollback capabilities for error recovery.
    Operations are split into storage and index components to enable fine-grained
    control and transaction-like behavior.

    The manager supports:
    1. Two-phase inserts (storage then indexes)
    2. Two-phase deletes (storage then indexes)
    3. Automatic rollback on failures
    4. Preimage capture for update operations

    Attributes:
        _logger: Logger instance for recording diagnostic information.
        _get_record: Function to retrieve a record by ID from storage.
    """
    def __init__(self, logger, get_record_func):
        self._logger = logger
        self._get_record = get_record_func

    def storage_insert(self, version, data_packet):
        """Perform only the storage part of an insert operation with rollback capability.

        This method handles writing data to storage and manages rollback in case of
        failures. It captures the preimage (previous state) of the record if it exists,
        which is returned to allow for index updates.

        Args:
            version: CollectionVersion object containing storage.
            data_packet: DataPacket object containing the record data to insert.

        Returns:
            DataPacket: The preimage (previous state) of the record, or a nonexistent
                       DataPacket if the record is new.

        Raises:
            StorageFailureException: If storage operations fail and cannot be rolled back.
            Exception: Any exception that occurs during storage operations.

        Note:
            If an exception occurs during storage operations, this method attempts to
            roll back the storage state before re-raising the exception.
        """
        preimage, old_loc = self._prepare_preimage(version, data_packet)
        new_loc = None
        try:
            new_loc = self._write_storage(version, data_packet)
            return preimage
        except Exception:
            self._logger.error(f"Storage insert failed, rolling back {data_packet.record_id}", exc_info=True)
            self._rollback_insert_storage(version, data_packet, old_loc, new_loc)
            raise

    def _prepare_preimage(self, version, data_packet):
        old_loc = version.storage.get_record_location(data_packet.record_id)
        if old_loc:
            self._logger.debug(f"Record {data_packet.record_id} exists, performing update")
            preimage = self._get_record(version, data_packet.record_id)
        else:
            self._logger.debug(f"Record {data_packet.record_id} is new")
            preimage = DataPacket.create_nonexistent(record_id=data_packet.record_id)
        return preimage, old_loc

    def index_insert(self, version, data_packet, preimage=None):
        """Perform only the index part of an insert operation with rollback capability.

        This method handles updating all indexes (metadata, document, vector) and manages
        rollback in case of failures. It uses different operations based on whether
        the record already exists.

        Args:
            version: CollectionVersion object containing the indexes.
            data_packet: DataPacket object containing the record data to index.
            preimage: Optional DataPacket representing the previous state of the record.
                     If not provided, it will be retrieved using the get_record function.

        Raises:
            Exception: Any exception that occurs during index operations.

        Note:
            - For new records, add operations are used on all indexes.
            - For existing records, update operations are used for metadata and document
              indexes, while the vector index uses delete-then-add for safety.
            - If an exception occurs during indexing, this method attempts to roll back
              the index state before re-raising the exception.
        """
        record_id = data_packet.record_id
        if preimage is None:
            try:
                preimage = self._get_record(version, record_id)
                if preimage is None:
                    preimage = DataPacket.create_nonexistent(record_id=record_id)
            except Exception:
                preimage = DataPacket.create_nonexistent(record_id=record_id)

        try:
            # Handle index updates differently based on whether record exists
            if preimage.is_nonexistent():
                # New record - use add operations
                self._logger.debug(f"Adding new record {record_id} to indexes")
                version.meta_index.add(data_packet.to_metadata_packet())
                version.doc_index.add(data_packet.to_document_packet())
                version.vec_index.add(data_packet.to_vector_packet())
            else:
                # Existing record - use update operations
                self._logger.debug(f"Updating existing record {record_id} in indexes")
                version.meta_index.update(preimage.to_metadata_packet(), data_packet.to_metadata_packet())
                version.doc_index.update(preimage.to_document_packet(), data_packet.to_document_packet())
                # For vector index, delete then add is safer than update
                version.vec_index.delete(record_id=record_id)
                version.vec_index.add(data_packet.to_vector_packet())
        except Exception:
            self._logger.error(f"Index insert failed, rolling back {data_packet.record_id}", exc_info=True)
            self._rollback_insert_index(version, data_packet, preimage)
            raise

    def _write_storage(self, version, data_packet):
        rec_bytes = data_packet.to_bytes()
        new_offset = version.storage.allocate(len(rec_bytes))
        loc = LocationPacket(
            record_id=data_packet.record_id,
            offset=new_offset,
            size=len(rec_bytes)
        )
        self._logger.debug(f"Writing record {data_packet.record_id} at offset {new_offset}")
        try:
            version.storage.write_and_index(rec_bytes, loc)
        except Exception as e:
            msg = f"Storage update failed for record {data_packet.record_id}"
            self._logger.debug(msg)
            raise StorageFailureException(msg, e)
        return loc

    def _rollback_insert_storage(self, version, data_packet, old_loc, new_loc):
        """Rollback only the storage side of an insert."""
        if new_loc:
            try:
                version.storage.mark_deleted(data_packet.record_id)
                version.storage.delete_record(data_packet.record_id)
                if old_loc:
                    old_loc.validate_checksum()
                    version.storage.add_record(old_loc)
            except Exception as e:
                self._logger.debug("Failed to rollback storage insert: %s", e)

    def _rollback_insert_index(self, version, data_packet, preimage):
        """Rollback only the index side of an insert (vector, meta, doc)."""
        # Rollback vector
        try:
            version.vec_index.delete(record_id=data_packet.record_id)
            if not preimage.is_nonexistent():
                version.vec_index.add(preimage.to_vector_packet())
        except Exception as e:
            self._logger.debug("Failed to rollback vector index: %s", e)

        # Rollback doc and meta
        for idx_name, fn in [
            ('doc_index', preimage.to_document_packet),
            ('meta_index', preimage.to_metadata_packet)
        ]:
            try:
                idx = getattr(version, idx_name)
                idx.delete(fn())
                if not preimage.is_nonexistent():
                    idx.add(fn())
            except Exception:
                self._logger.debug(f"Failed to rollback {idx_name}")

    def storage_delete(self, version, data_packet):
        """Perform only the storage part of a delete operation with rollback capability.

        This method handles removing data from storage and manages rollback in case of
        failures. It captures the preimage (state before deletion) of the record,
        which is returned to allow for index updates.

        Args:
            version: CollectionVersion object containing storage.
            data_packet: DataPacket object containing the record ID to delete.

        Returns:
            DataPacket: The preimage (state before deletion) of the record, or a nonexistent
                       DataPacket if the record didn't exist.

        Raises:
            StorageFailureException: If storage operations fail and cannot be rolled back.

        Note:
            If an exception occurs during storage operations, this method attempts to
            roll back the storage state before re-raising the exception.
        """
        record_id = data_packet.record_id
        self._logger.debug(f"Applying storage delete for record {record_id}")

        # A) load old location & preimage
        old_loc = version.storage.get_record_location(record_id)
        if not old_loc:
            self._logger.warning(f"Attempted to delete non-existent record {record_id}")
            return DataPacket.create_nonexistent(record_id=record_id)
        preimage = self._get_record(version, record_id)

        removed_storage = False
        try:
            # mark deleted and remove from storage index
            self._logger.debug(f"Marking record {record_id} as deleted in storage")
            version.storage.mark_deleted(record_id)
            removed_storage = True

            self._logger.debug(f"Removing record {record_id} from storage")
            version.storage.delete_record(record_id)

            return preimage

        except Exception as e:
            self._logger.error(f"Storage removal failed for record {record_id}, rolling back", exc_info=True)
            # rollback storage deletion
            if removed_storage:
                try:
                    version.storage.add_record(
                        LocationPacket(record_id=record_id, offset=old_loc.offset, size=old_loc.size)
                    )
                except Exception:
                    self._logger.debug("Failed to rollback storage delete")
            raise StorageFailureException(f"Storage removal failed for record {record_id}", e)

    def index_delete(self, version, data_packet, preimage):
        """Perform only the index part of a delete operation with rollback capability.

        This method handles removing the record from all indexes (metadata, document, vector)
        and manages rollback in case of failures.

        Args:
            version: CollectionVersion object containing the indexes.
            data_packet: DataPacket object containing the record ID to delete.
            preimage: DataPacket representing the state of the record before deletion,
                     typically obtained from storage_delete().

        Raises:
            Exception: Any exception that occurs during index operations.

        Note:
            - If the preimage indicates the record doesn't exist, this method does nothing.
            - Operations are executed in order: metadata index, document index, vector index.
            - If an exception occurs during indexing, this method attempts to roll back
              the executed operations in reverse order before re-raising the exception.
        """
        record_id = data_packet.record_id
        self._logger.debug(f"Applying index delete for record {record_id}")

        # Skip if record doesn't exist in indexes
        if preimage.is_nonexistent():
            self._logger.debug(
                f"Record {record_id} doesn't exist in indexes, skipping delete operation"
            )
            return

        # Prepare operations for delete and rollback
        operations = [
            {
                'name': 'metadata',
                'delete': lambda: version.meta_index.delete(preimage.to_metadata_packet()),
                'add': lambda: version.meta_index.add(preimage.to_metadata_packet())
            },
            {
                'name': 'document',
                'delete': lambda: version.doc_index.delete(preimage.to_document_packet()),
                'add': lambda: version.doc_index.add(preimage.to_document_packet())
            },
            {
                'name': 'vector',
                'delete': lambda: version.vec_index.delete(record_id=record_id),
                'add': lambda: version.vec_index.add(preimage.to_vector_packet())
            }
        ]

        executed = []
        try:
            for op in operations:
                self._logger.debug(
                    f"Removing record {record_id} from {op['name']} index"
                )
                op['delete']()
                executed.append(op)
        except Exception:
            self._logger.error(
                f"Index removal failed for record {record_id}, rolling back",
                exc_info=True
            )
            # Rollback in reverse order
            for op in reversed(executed):
                try:
                    op['add']()
                except Exception:
                    self._logger.debug(
                        f"Failed to rollback {op['name']} index delete"
                    )
            raise
