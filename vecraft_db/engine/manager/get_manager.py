from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.index_packets import LocationPacket
from vecraft_db.core.lock.mvcc_manager import CollectionVersion
from vecraft_exception_model.exception import ChecksumValidationFailureError


class GetManager:
    """Manager for retrieving records from storage with validation.

    This class provides a method to retrieve records from storage, performing
    validation checks to ensure data integrity. It handles cases where records
    don't exist and validates that retrieved records match the requested IDs.

    Attributes:
        _logger: Logger instance for recording diagnostic information.
    """
    def __init__(self, logger):
        self._logger = logger

    def get(self, version: CollectionVersion, record_id: str) -> DataPacket:
        """Retrieve a record from storage with validation.

        This method retrieves a record from storage based on its ID and performs
        validation to ensure data integrity. If the record doesn't exist or can't
        be read, a nonexistent DataPacket is returned.

        The method performs these steps:
        1. Get the record location from storage
        2. If location doesn't exist, return a nonexistent DataPacket
        3. Read the data from storage using location information
        4. If data can't be read, return a nonexistent DataPacket
        5. Convert the binary data to a DataPacket
        6. Verify that the returned record ID matches the requested record ID

        Args:
            version: CollectionVersion object that contains storage and other
                    collection-related components.
            record_id: String identifier of the record to retrieve.

        Returns:
            DataPacket: The retrieved and validated record data.
                       If the record doesn't exist, returns a nonexistent DataPacket.

        Raises:
            ChecksumValidationFailureError: If the record ID in the retrieved data
                                           doesn't match the requested record ID,
                                           indicating potential data corruption.
        """
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