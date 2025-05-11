import unittest
from unittest.mock import patch

import numpy as np

from src.vecraft_api.rest.data_model_utils import DataModelUtils
from src.vecraft_api.rest.data_models import DataPacketModel, NumpyArray, QueryPacketModel
from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.query_packet import QueryPacket


class TestDataModelUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create real numpy array data
        self.test_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.test_query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Create real metadata
        self.test_metadata = {
            "timestamp": "2024-01-01T00:00:00",
            "source": "test_source",
            "version": "1.0"
        }

        # Create real where conditions
        self.test_where = {"category": "test", "active": True}
        self.test_where_document = {"content": "test document"}

        # Common test data
        self.test_record_id = "record_123"
        self.test_original_data = {"content": "This is test content", "id": 123}
        self.test_checksum_algorithm = "sha256"
        self.test_k = 10

    def create_data_packet_model(self, include_vector=True):
        """Helper to create a DataPacketModel instance."""
        vector = NumpyArray.from_numpy(self.test_vector) if include_vector else None
        return DataPacketModel(
            type="RECORD",
            record_id=self.test_record_id,
            original_data=self.test_original_data,
            metadata=self.test_metadata,
            checksum_algorithm=self.test_checksum_algorithm,
            vector=vector,
            checksum=None  # Will be computed by DataPacket
        )

    def create_query_packet_model(self):
        """Helper to create a QueryPacketModel instance."""
        return QueryPacketModel(
            query_vector=NumpyArray.from_numpy(self.test_query_vector),
            k=self.test_k,
            where=self.test_where,
            where_document=self.test_where_document,
            checksum_algorithm=self.test_checksum_algorithm,
            checksum=None  # Will be computed by QueryPacket
        )

    def test_convert_to_data_packet_with_vector(self):
        """Test converting DataPacketModel to DataPacket with vector."""
        # Create a real DataPacketModel
        model = self.create_data_packet_model(include_vector=True)

        # Convert to DataPacket
        packet = DataModelUtils.convert_to_data_packet(model)

        # Assertions
        self.assertIsInstance(packet, DataPacket)
        self.assertEqual(packet.record_id, self.test_record_id)
        self.assertEqual(packet.original_data, self.test_original_data)
        self.assertEqual(packet.metadata, self.test_metadata)
        self.assertEqual(packet.checksum_algorithm, self.test_checksum_algorithm)
        np.testing.assert_array_equal(packet.vector, self.test_vector)

        # Verify checksum is computed
        self.assertIsNotNone(packet.checksum)

    def test_convert_from_data_packet_with_vector(self):
        """Test converting DataPacket to DataPacketModel with vector."""
        # Create a real DataPacket
        packet = DataPacket.create_record(
            record_id=self.test_record_id,
            original_data=self.test_original_data,
            metadata=self.test_metadata,
            checksum_algorithm=self.test_checksum_algorithm,
            vector=self.test_vector
        )

        # Convert to DataPacketModel
        model = DataModelUtils.convert_from_data_packet(packet)

        # Assertions
        self.assertIsInstance(model, DataPacketModel)
        self.assertEqual(model.record_id, self.test_record_id)
        self.assertEqual(model.original_data, self.test_original_data)
        self.assertEqual(model.metadata, self.test_metadata)
        self.assertEqual(model.checksum_algorithm, self.test_checksum_algorithm)
        self.assertIsNotNone(model.vector)
        np.testing.assert_array_equal(model.vector.to_numpy(), self.test_vector)
        self.assertEqual(model.checksum, packet.checksum)

    def test_convert_to_query_packet(self):
        """Test converting QueryPacketModel to QueryPacket."""
        # Create a real QueryPacketModel
        model = self.create_query_packet_model()

        # Convert to QueryPacket
        packet = DataModelUtils.convert_to_query_packet(model)

        # Assertions
        self.assertIsInstance(packet, QueryPacket)
        np.testing.assert_array_equal(packet.query_vector, self.test_query_vector)
        self.assertEqual(packet.k, self.test_k)
        self.assertEqual(packet.where, self.test_where)
        self.assertEqual(packet.where_document, self.test_where_document)
        self.assertEqual(packet.checksum_algorithm, self.test_checksum_algorithm)

        # Verify checksum is computed
        self.assertIsNotNone(packet.checksum)

    def test_convert_from_query_packet(self):
        """Test converting QueryPacket to QueryPacketModel."""
        # Create a real QueryPacket
        packet = QueryPacket(
            query_vector=self.test_query_vector,
            k=self.test_k,
            where=self.test_where,
            where_document=self.test_where_document,
            checksum_algorithm=self.test_checksum_algorithm
        )

        # Convert to QueryPacketModel
        model = DataModelUtils.convert_from_query_packet(packet)

        # Assertions
        self.assertIsInstance(model, QueryPacketModel)
        np.testing.assert_array_equal(model.query_vector.to_numpy(), self.test_query_vector)
        self.assertEqual(model.k, self.test_k)
        self.assertEqual(model.where, self.test_where)
        self.assertEqual(model.where_document, self.test_where_document)
        self.assertEqual(model.checksum_algorithm, self.test_checksum_algorithm)
        self.assertEqual(model.checksum, packet.checksum)

    def test_round_trip_data_packet_conversion(self):
        """Test that converting to and from DataPacket preserves data."""
        # Start with a DataPacketModel
        original_model = self.create_data_packet_model(include_vector=True)

        # Convert to DataPacket and back
        packet = DataModelUtils.convert_to_data_packet(original_model)
        converted_model = DataModelUtils.convert_from_data_packet(packet)

        # Compare original and converted models
        self.assertEqual(original_model.record_id, converted_model.record_id)
        self.assertEqual(original_model.original_data, converted_model.original_data)
        self.assertEqual(original_model.metadata, converted_model.metadata)
        self.assertEqual(original_model.checksum_algorithm, converted_model.checksum_algorithm)
        np.testing.assert_array_equal(
            original_model.vector.to_numpy(),
            converted_model.vector.to_numpy()
        )

    def test_round_trip_query_packet_conversion(self):
        """Test that converting to and from QueryPacket preserves data."""
        # Start with a QueryPacketModel
        original_model = self.create_query_packet_model()

        # Convert to QueryPacket and back
        packet = DataModelUtils.convert_to_query_packet(original_model)
        converted_model = DataModelUtils.convert_from_query_packet(packet)

        # Compare original and converted models
        np.testing.assert_array_equal(
            original_model.query_vector.to_numpy(),
            converted_model.query_vector.to_numpy()
        )
        self.assertEqual(original_model.k, converted_model.k)
        self.assertEqual(original_model.where, converted_model.where)
        self.assertEqual(original_model.where_document, converted_model.where_document)
        self.assertEqual(original_model.checksum_algorithm, converted_model.checksum_algorithm)

    @patch.object(DataPacket, 'validate_checksum')
    def test_convert_to_data_packet_checksum_validation_error(self, mock_validate):
        """Test error handling during checksum validation in convert_to_data_packet."""
        # Setup mock to raise exception
        mock_validate.side_effect = ValueError("Checksum validation failed")

        # Create a DataPacketModel
        model = self.create_data_packet_model(include_vector=True)

        # Attempt conversion and expect exception
        with self.assertRaises(ValueError) as context:
            DataModelUtils.convert_to_data_packet(model)

        self.assertEqual(str(context.exception), "Checksum validation failed")

    @patch.object(DataPacket, 'validate_checksum')
    def test_convert_from_data_packet_checksum_validation_error(self, mock_validate):
        """Test error handling during checksum validation in convert_from_data_packet."""
        # Create a real DataPacket
        packet = DataPacket.create_record(
            record_id=self.test_record_id,
            original_data=self.test_original_data,
            metadata=self.test_metadata,
            checksum_algorithm=self.test_checksum_algorithm,
            vector=self.test_vector
        )

        # Setup mock to raise exception
        mock_validate.side_effect = ValueError("Checksum validation failed")

        # Attempt conversion and expect exception
        with self.assertRaises(ValueError) as context:
            DataModelUtils.convert_from_data_packet(packet)

        self.assertEqual(str(context.exception), "Checksum validation failed")

    @patch.object(QueryPacket, 'validate_checksum')
    def test_convert_to_query_packet_checksum_validation_error(self, mock_validate):
        """Test error handling during checksum validation in convert_to_query_packet."""
        # Setup mock to raise exception
        mock_validate.side_effect = ValueError("Checksum validation failed")

        # Create a QueryPacketModel
        model = self.create_query_packet_model()

        # Attempt conversion and expect exception
        with self.assertRaises(ValueError) as context:
            DataModelUtils.convert_to_query_packet(model)

        self.assertEqual(str(context.exception), "Checksum validation failed")

    @patch.object(QueryPacket, 'validate_checksum')
    def test_convert_from_query_packet_checksum_validation_error(self, mock_validate):
        """Test error handling during checksum validation in convert_from_query_packet."""
        # Create a real QueryPacket
        packet = QueryPacket(
            query_vector=self.test_query_vector,
            k=self.test_k,
            where=self.test_where,
            where_document=self.test_where_document,
            checksum_algorithm=self.test_checksum_algorithm
        )

        # Setup mock to raise exception
        mock_validate.side_effect = ValueError("Checksum validation failed")

        # Attempt conversion and expect exception
        with self.assertRaises(ValueError) as context:
            DataModelUtils.convert_from_query_packet(packet)

        self.assertEqual(str(context.exception), "Checksum validation failed")

    def test_empty_metadata_and_where_conditions(self):
        """Test handling of empty metadata and where conditions."""
        # Create DataPacket with empty metadata
        packet = DataPacket.create_record(
            record_id=self.test_record_id,
            original_data=self.test_original_data,
            metadata={},
            checksum_algorithm=self.test_checksum_algorithm,
            vector=self.test_vector
        )

        # Convert to model
        model = DataModelUtils.convert_from_data_packet(packet)
        self.assertEqual(model.metadata, {})

        # Create QueryPacket with empty where conditions
        query_packet = QueryPacket(
            query_vector=self.test_query_vector,
            k=self.test_k,
            where={},
            where_document={},
            checksum_algorithm=self.test_checksum_algorithm
        )

        # Convert to model
        query_model = DataModelUtils.convert_from_query_packet(query_packet)
        self.assertEqual(query_model.where, {})
        self.assertEqual(query_model.where_document, {})

    def test_large_vector_conversion(self):
        """Test conversion with large vectors."""
        # Create a large vector
        rng = np.random.default_rng(42)
        large_vector = rng.random(1000)

        # Create DataPacketModel with large vector
        model = DataPacketModel(
            type="RECORD",
            record_id=self.test_record_id,
            original_data=self.test_original_data,
            metadata=self.test_metadata,
            checksum_algorithm=self.test_checksum_algorithm,
            vector=NumpyArray.from_numpy(large_vector),
            checksum=None
        )

        # Convert to DataPacket and back
        packet = DataModelUtils.convert_to_data_packet(model)
        converted_model = DataModelUtils.convert_from_data_packet(packet)

        # Verify large vector is preserved
        np.testing.assert_array_almost_equal(
            large_vector,
            converted_model.vector.to_numpy()
        )

    def test_tombstone_conversion(self):
        """Test conversion with tombstone."""
        model = DataPacketModel(
            type="TOMBSTONE",
            record_id=self.test_record_id,
            original_data=None,
            metadata=None,
            vector=None,
            checksum_algorithm=self.test_checksum_algorithm,
            checksum=None
        )

        # Convert to DataPacket and back
        packet = DataModelUtils.convert_to_data_packet(model)
        converted_model = DataModelUtils.convert_from_data_packet(packet)

        # Verify converted packet is tombstone
        self.assertTrue(packet.is_tombstone())
        self.assertEqual("TOMBSTONE", converted_model.type)

    def test_nonexistent_conversion(self):
        """Test conversion with nonexistent."""
        model = DataPacketModel(
            type="NONEXISTENT",
            record_id=self.test_record_id,
            original_data=None,
            metadata=None,
            vector=None,
            checksum_algorithm=self.test_checksum_algorithm,
            checksum=None
        )

        # Convert to DataPacket and back
        packet = DataModelUtils.convert_to_data_packet(model)
        converted_model = DataModelUtils.convert_from_data_packet(packet)

        # Verify converted packet is nonexistent
        self.assertTrue(packet.is_nonexistent())
        self.assertEqual("NONEXISTENT", converted_model.type)


if __name__ == '__main__':
    unittest.main()