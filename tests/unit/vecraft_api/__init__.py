import unittest
from unittest.mock import Mock, patch
import numpy as np

from src.vecraft_api.rest.data_model_utils import DataModelUtils
from src.vecraft_api.rest.data_models import DataPacketModel, NumpyArray, QueryPacketModel
from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.query_packet import QueryPacket


class TestDataModelUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock numpy array data
        self.test_vector = np.array([1.0, 2.0, 3.0])
        self.test_query_vector = np.array([4.0, 5.0, 6.0])

        # Create mock metadata
        self.test_metadata = {
            "key1": "value1",
            "key2": "value2"
        }

        # Create mock where conditions
        self.test_where = {"field": "value"}
        self.test_where_document = {"doc_field": "doc_value"}

    def test_convert_to_data_packet_with_vector(self):
        """Test converting DataPacketModel to DataPacket with vector."""
        # Create mock NumpyArray
        mock_numpy_array = Mock(spec=NumpyArray)
        mock_numpy_array.to_numpy.return_value = self.test_vector

        # Create mock DataPacketModel
        mock_model = Mock(spec=DataPacketModel)
        mock_model.record_id = "test_id"
        mock_model.original_data = {"test": "data"}
        mock_model.metadata = self.test_metadata
        mock_model.checksum_algorithm = "sha256"
        mock_model.vector = mock_numpy_array

        # Mock DataPacket
        with patch('your_module.DataPacket') as MockDataPacket:
            mock_packet_instance = Mock(spec=DataPacket)
            MockDataPacket.return_value = mock_packet_instance

            # Call the method
            result = DataModelUtils.convert_to_data_packet(mock_model)

            # Assertions
            MockDataPacket.assert_called_once_with(
                record_id="test_id",
                original_data={"test": "data"},
                metadata=self.test_metadata,
                checksum_algorithm="sha256",
                vector=self.test_vector
            )
            mock_numpy_array.to_numpy.assert_called_once()
            mock_packet_instance.validate_checksum.assert_called_once()
            self.assertEqual(result, mock_packet_instance)

    def test_convert_to_data_packet_without_vector(self):
        """Test converting DataPacketModel to DataPacket without vector."""
        # Create mock DataPacketModel without vector
        mock_model = Mock(spec=DataPacketModel)
        mock_model.record_id = "test_id"
        mock_model.original_data = {"test": "data"}
        mock_model.metadata = self.test_metadata
        mock_model.checksum_algorithm = "sha256"
        mock_model.vector = None

        # Mock DataPacket
        with patch('your_module.DataPacket') as MockDataPacket:
            mock_packet_instance = Mock(spec=DataPacket)
            MockDataPacket.return_value = mock_packet_instance

            # Call the method
            result = DataModelUtils.convert_to_data_packet(mock_model)

            # Assertions
            MockDataPacket.assert_called_once_with(
                record_id="test_id",
                original_data={"test": "data"},
                metadata=self.test_metadata,
                checksum_algorithm="sha256"
            )
            mock_packet_instance.validate_checksum.assert_called_once()
            self.assertEqual(result, mock_packet_instance)

    def test_convert_from_data_packet_with_vector(self):
        """Test converting DataPacket to DataPacketModel with vector."""
        # Create mock DataPacket
        mock_packet = Mock(spec=DataPacket)
        mock_packet.vector = self.test_vector
        mock_packet.to_dict.return_value = {
            "record_id": "test_id",
            "original_data": {"test": "data"},
            "metadata": self.test_metadata,
            "checksum_algorithm": "sha256",
            "checksum": "test_checksum",
            "vector": self.test_vector
        }

        # Mock NumpyArray.from_numpy
        with patch.object(NumpyArray, 'from_numpy') as mock_from_numpy:
            mock_numpy_array = Mock(spec=NumpyArray)
            mock_from_numpy.return_value = mock_numpy_array

            # Mock DataPacketModel
            with patch('your_module.DataPacketModel') as MockDataPacketModel:
                mock_model_instance = Mock(spec=DataPacketModel)
                MockDataPacketModel.return_value = mock_model_instance

                # Call the method
                result = DataModelUtils.convert_from_data_packet(mock_packet)

                # Assertions
                mock_packet.validate_checksum.assert_called_once()
                mock_packet.to_dict.assert_called_once()
                mock_from_numpy.assert_called_once_with(self.test_vector)

                # Check that DataPacketModel was called with correct arguments
                expected_call_args = {
                    "record_id": "test_id",
                    "original_data": {"test": "data"},
                    "metadata": self.test_metadata,
                    "checksum_algorithm": "sha256",
                    "checksum": "test_checksum",
                    "vector": mock_numpy_array
                }
                MockDataPacketModel.assert_called_once_with(**expected_call_args)
                self.assertEqual(result, mock_model_instance)

    def test_convert_from_data_packet_without_vector(self):
        """Test converting DataPacket to DataPacketModel without vector."""
        # Create mock DataPacket without vector
        mock_packet = Mock(spec=DataPacket)
        mock_packet.vector = None
        mock_packet.to_dict.return_value = {
            "record_id": "test_id",
            "original_data": {"test": "data"},
            "metadata": self.test_metadata,
            "checksum_algorithm": "sha256",
            "checksum": "test_checksum"
        }

        # Mock DataPacketModel
        with patch('your_module.DataPacketModel') as MockDataPacketModel:
            mock_model_instance = Mock(spec=DataPacketModel)
            MockDataPacketModel.return_value = mock_model_instance

            # Call the method
            result = DataModelUtils.convert_from_data_packet(mock_packet)

            # Assertions
            mock_packet.validate_checksum.assert_called_once()
            mock_packet.to_dict.assert_called_once()

            # Check that DataPacketModel was called with correct arguments
            MockDataPacketModel.assert_called_once_with(
                record_id="test_id",
                original_data={"test": "data"},
                metadata=self.test_metadata,
                checksum_algorithm="sha256",
                checksum="test_checksum"
            )
            self.assertEqual(result, mock_model_instance)

    def test_convert_to_query_packet(self):
        """Test converting QueryPacketModel to QueryPacket."""
        # Create mock NumpyArray
        mock_numpy_array = Mock(spec=NumpyArray)
        mock_numpy_array.to_numpy.return_value = self.test_query_vector

        # Create mock QueryPacketModel
        mock_model = Mock(spec=QueryPacketModel)
        mock_model.query_vector = mock_numpy_array
        mock_model.k = 10
        mock_model.where = self.test_where
        mock_model.where_document = self.test_where_document
        mock_model.checksum_algorithm = "sha256"

        # Mock QueryPacket
        with patch('your_module.QueryPacket') as MockQueryPacket:
            mock_packet_instance = Mock(spec=QueryPacket)
            MockQueryPacket.return_value = mock_packet_instance

            # Call the method
            result = DataModelUtils.convert_to_query_packet(mock_model)

            # Assertions
            MockQueryPacket.assert_called_once_with(
                query_vector=self.test_query_vector,
                k=10,
                where=self.test_where,
                where_document=self.test_where_document,
                checksum_algorithm="sha256"
            )
            mock_numpy_array.to_numpy.assert_called_once()
            mock_packet_instance.validate_checksum.assert_called_once()
            self.assertEqual(result, mock_packet_instance)

    def test_convert_from_query_packet(self):
        """Test converting QueryPacket to QueryPacketModel."""
        # Create mock QueryPacket
        mock_packet = Mock(spec=QueryPacket)
        mock_packet.query_vector = self.test_query_vector
        mock_packet.checksum = "test_checksum"
        mock_packet.to_dict.return_value = {
            "query_vector": self.test_query_vector,
            "k": 10,
            "where": self.test_where,
            "where_document": self.test_where_document,
            "checksum_algorithm": "sha256"
        }

        # Mock NumpyArray.from_numpy
        with patch.object(NumpyArray, 'from_numpy') as mock_from_numpy:
            mock_numpy_array = Mock(spec=NumpyArray)
            mock_from_numpy.return_value = mock_numpy_array

            # Mock QueryPacketModel
            with patch('your_module.QueryPacketModel') as MockQueryPacketModel:
                mock_model_instance = Mock(spec=QueryPacketModel)
                MockQueryPacketModel.return_value = mock_model_instance

                # Call the method
                result = DataModelUtils.convert_from_query_packet(mock_packet)

                # Assertions
                mock_packet.validate_checksum.assert_called_once()
                mock_packet.to_dict.assert_called_once()
                mock_from_numpy.assert_called_once_with(self.test_query_vector)

                # Check that QueryPacketModel was called with correct arguments
                expected_call_args = {
                    "query_vector": mock_numpy_array,
                    "k": 10,
                    "where": self.test_where,
                    "where_document": self.test_where_document,
                    "checksum_algorithm": "sha256",
                    "checksum": "test_checksum"
                }
                MockQueryPacketModel.assert_called_once_with(**expected_call_args)
                self.assertEqual(result, mock_model_instance)

    def test_convert_to_data_packet_checksum_validation_error(self):
        """Test error handling during checksum validation in convert_to_data_packet."""
        # Create mock DataPacketModel
        mock_model = Mock(spec=DataPacketModel)
        mock_model.record_id = "test_id"
        mock_model.original_data = {"test": "data"}
        mock_model.metadata = self.test_metadata
        mock_model.checksum_algorithm = "sha256"
        mock_model.vector = None

        # Mock DataPacket to raise an exception on validate_checksum
        with patch('your_module.DataPacket') as MockDataPacket:
            mock_packet_instance = Mock(spec=DataPacket)
            mock_packet_instance.validate_checksum.side_effect = ValueError("Checksum validation failed")
            MockDataPacket.return_value = mock_packet_instance

            # Call the method and expect exception
            with self.assertRaises(ValueError) as context:
                DataModelUtils.convert_to_data_packet(mock_model)

            self.assertEqual(str(context.exception), "Checksum validation failed")

    def test_convert_from_data_packet_checksum_validation_error(self):
        """Test error handling during checksum validation in convert_from_data_packet."""
        # Create mock DataPacket
        mock_packet = Mock(spec=DataPacket)
        mock_packet.validate_checksum.side_effect = ValueError("Checksum validation failed")

        # Call the method and expect exception
        with self.assertRaises(ValueError) as context:
            DataModelUtils.convert_from_data_packet(mock_packet)

        self.assertEqual(str(context.exception), "Checksum validation failed")
        mock_packet.validate_checksum.assert_called_once()

    def test_convert_to_query_packet_checksum_validation_error(self):
        """Test error handling during checksum validation in convert_to_query_packet."""
        # Create mock NumpyArray
        mock_numpy_array = Mock(spec=NumpyArray)
        mock_numpy_array.to_numpy.return_value = self.test_query_vector

        # Create mock QueryPacketModel
        mock_model = Mock(spec=QueryPacketModel)
        mock_model.query_vector = mock_numpy_array
        mock_model.k = 10
        mock_model.where = self.test_where
        mock_model.where_document = self.test_where_document
        mock_model.checksum_algorithm = "sha256"

        # Mock QueryPacket to raise an exception on validate_checksum
        with patch('your_module.QueryPacket') as MockQueryPacket:
            mock_packet_instance = Mock(spec=QueryPacket)
            mock_packet_instance.validate_checksum.side_effect = ValueError("Checksum validation failed")
            MockQueryPacket.return_value = mock_packet_instance

            # Call the method and expect exception
            with self.assertRaises(ValueError) as context:
                DataModelUtils.convert_to_query_packet(mock_model)

            self.assertEqual(str(context.exception), "Checksum validation failed")

    def test_convert_from_query_packet_checksum_validation_error(self):
        """Test error handling during checksum validation in convert_from_query_packet."""
        # Create mock QueryPacket
        mock_packet = Mock(spec=QueryPacket)
        mock_packet.validate_checksum.side_effect = ValueError("Checksum validation failed")

        # Call the method and expect exception
        with self.assertRaises(ValueError) as context:
            DataModelUtils.convert_from_query_packet(mock_packet)

        self.assertEqual(str(context.exception), "Checksum validation failed")
        mock_packet.validate_checksum.assert_called_once()


if __name__ == '__main__':
    unittest.main()