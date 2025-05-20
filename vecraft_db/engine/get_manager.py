from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.index_packets import LocationPacket
from vecraft_db.core.lock.mvcc_manager import CollectionVersion
from vecraft_exception_model.exception import ChecksumValidationFailureError


class GetManager:
    def __init__(self, logger):
        self._logger = logger

    def get(self, version: CollectionVersion, record_id: str) -> DataPacket:
        """Get a record from storage with validation"""
        loc = version.storage.get_record_location(record_id)
        if not loc:
            return DataPacket.create_nonexistent(record_id=record_id)

        data = version.storage.read(LocationPacket(record_id=record_id, offset=loc.offset, size=loc.size))
        if not data:
            return DataPacket.create_nonexistent(record_id=record_id)

        data_packet = DataPacket.from_bytes(data)

        # Verify that the returned record is the one that we request
        if data_packet.record_id != record_id:
            error_message = f"Returned record {data_packet.record_id} does not match expected record {record_id}"
            self._logger.error(error_message)
            raise ChecksumValidationFailureError(error_message)

        return data_packet