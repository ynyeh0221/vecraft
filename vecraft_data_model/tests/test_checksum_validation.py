import unittest

import numpy as np

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.index_packets import VectorPacket, MetadataPacket
from vecraft_data_model.query_packet import QueryPacket
from vecraft_exception_model.exception import ChecksumValidationFailureError


class ChecksumValidationTests(unittest.TestCase):

    def test_data_packet_checksum_validation_failure(self):
        """Test that DataPacket throws ChecksumValidationFailureError when checksum validation fails."""
        # Create a valid DataPacket
        data_packet = DataPacket.create_record(
            record_id="test1",
            original_data={"text": "test data"},
            vector=np.array([0.1, 0.2, 0.3, 0.4]),
            metadata={"tags": ["test"]}
        )

        # Verify initial checksum is valid
        self.assertTrue(data_packet.validate_checksum())

        # Tamper with the data without updating checksum
        data_packet.original_data = {"text": "tampered data"}

        # Validation should now fail with exception
        with self.assertRaises(ChecksumValidationFailureError) as context:
            data_packet.validate_checksum()

        # Verify exception contains correct record_id
        self.assertEqual(context.exception.record_id, "test1")
        self.assertIn("DataPacket checksum validation failed", str(context.exception))

    def test_query_packet_checksum_validation_failure(self):
        """Test that QueryPacket throws ChecksumValidationFailureError when checksum validation fails."""
        # Create a valid QueryPacket
        query_packet = QueryPacket(
            query_vector=np.array([0.1, 0.2, 0.3, 0.4]),
            k=5,
            where={"tags": ["test"]}
        )

        # Verify initial checksum is valid
        self.assertTrue(query_packet.validate_checksum())

        # Tamper with the query vector without updating checksum
        query_packet.query_vector = np.array([0.5, 0.6, 0.7, 0.8])

        # Validation should now fail with exception
        with self.assertRaises(ChecksumValidationFailureError) as context:
            query_packet.validate_checksum()

        self.assertIn("QueryPacket checksum validation failed", str(context.exception))

    def test_index_item_checksum_validation_failure(self):
        """Test that IndexItem throws ChecksumValidationFailureError when checksum validation fails."""
        # Create a valid IndexItem
        index_item = VectorPacket(
            record_id="test1",
            vector=np.array([0.1, 0.2, 0.3, 0.4])
        )

        # Verify initial checksum is valid
        self.assertTrue(index_item.validate_checksum())

        # Tamper with the document without updating checksum
        index_item.vector = np.array([0.1, 0.2, 0.3, 0.5])

        # Validation should now fail with exception
        with self.assertRaises(ChecksumValidationFailureError) as context:
            index_item.validate_checksum()

        # Verify exception contains correct record_id
        self.assertEqual(context.exception.record_id, "test1")
        self.assertIn("IndexItem checksum validation failed", str(context.exception))

    def test_metadata_item_checksum_validation_failure(self):
        """Test that MetadataItem throws ChecksumValidationFailureError when checksum validation fails."""
        # Create a valid MetadataItem
        metadata_item = MetadataPacket(
            record_id="test1",
            metadata={"tags": ["test"]}
        )

        # Verify initial checksum is valid
        self.assertTrue(metadata_item.validate_checksum())

        # Tamper with the metadata without updating checksum
        metadata_item.metadata = {"tags": ["tampered"]}

        # Validation should now fail with exception
        with self.assertRaises(ChecksumValidationFailureError) as context:
            metadata_item.validate_checksum()

        # Verify exception contains correct record_id
        self.assertEqual(context.exception.record_id, "test1")
        self.assertIn("MetadataItem checksum validation failed", str(context.exception))


if __name__ == '__main__':
    unittest.main()