import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST

from vecraft_data_model.data_model_utils import DataModelUtils
from vecraft_data_model.data_models import DataPacketModel, NumpyArray, QueryPacketModel, InsertRequest, SearchRequest
from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.search_data_packet import SearchDataPacket
from vecraft_db.rest.vecraft_rest_api_server import VecraftRestAPI
from vecraft_exception_model.exception import ChecksumValidationFailureError, RecordNotFoundError


class TestVecraftRestAPI(unittest.TestCase):
    def setUp(self):
        from prometheus_client import REGISTRY
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            REGISTRY.unregister(collector)

        """Set up test fixtures before each test method."""
        # Create test data
        self.test_collection = "test_collection"
        self.test_record_id = "record_123"

        # Setup test vectors
        rng = np.random.default_rng(42)  # Use seeded generator for reproducibility
        self.test_vector = rng.random(5)
        self.test_query_vector = rng.random(5)

        # Create test data
        self.test_original_data = {"content": "Test content", "id": 123}
        self.test_metadata = {"source": "test", "timestamp": "2024-01-01"}
        self.test_checksum_algorithm = "sha256"

        # Create a mock for VecraftClient
        self.mock_client_patcher = patch('vecraft_db.client.k8s_aware_vecraft_client.K8sAwareVecraftClient')
        self.mock_client_class = self.mock_client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client

        # Initialize the API with the mocked client
        self.api = VecraftRestAPI(root="/tmp/vecraft") # NOSONAR

        # Replace the client with our controlled mock
        self.api.client = self.mock_client

        # Create a test client
        self.client = TestClient(self.api.app)

    def tearDown(self):
        """Clean up after each test method."""
        self.mock_client_patcher.stop()

        # Clean up prometheus registry
        from prometheus_client import REGISTRY
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            REGISTRY.unregister(collector)

    def create_test_data_packet_model(self):
        """Helper to create a test DataPacketModel."""
        return DataPacketModel(
            type="RECORD",
            record_id=self.test_record_id,
            original_data=self.test_original_data,
            metadata=self.test_metadata,
            checksum_algorithm=self.test_checksum_algorithm,
            vector=NumpyArray.from_numpy(self.test_vector),
            checksum=None  # Let DataPacket compute this
        )

    def create_test_data_packet(self):
        """Helper to create a test DataPacket."""
        return DataPacket.create_record(
            record_id=self.test_record_id,
            original_data=self.test_original_data,
            metadata=self.test_metadata,
            checksum_algorithm=self.test_checksum_algorithm,
            vector=self.test_vector
        )

    def create_test_query_packet_model(self):
        """Helper to create a test QueryPacketModel."""
        return QueryPacketModel(
            query_vector=NumpyArray.from_numpy(self.test_query_vector),
            k=10,
            where={"category": "test"},
            where_document={"content": "test"},
            checksum_algorithm=self.test_checksum_algorithm,
            checksum=None  # Let QueryPacket compute this
        )

    def test_initialization(self):
        """Test the initialization of VecraftAPI."""
        # Check if routes are set up
        self.assertIn("/collections/{collection}/insert", [route.path for route in self.api.app.routes])
        self.assertIn("/collections/{collection}/search", [route.path for route in self.api.app.routes])
        self.assertIn("/collections/{collection}/records/{record_id}", [route.path for route in self.api.app.routes])

    def test_insert_route(self):
        """Test the insert route."""
        # Setup test data
        test_model = self.create_test_data_packet_model()

        # Create a real DataPacket for the return value using the real DataModelUtils
        expected_packet = DataModelUtils.convert_to_data_packet(test_model)
        self.mock_client.insert.return_value = expected_packet

        # Create an InsertRequest instance instead of using a raw dict
        insert_request = InsertRequest(packet=test_model)

        # Make the request with proper serialization
        response = self.client.post(
            f"/collections/{self.test_collection}/insert",
            json=insert_request.dict(by_alias=True, exclude_none=True)
        )

        # Check response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["record_id"], self.test_record_id)

        # Verify method calls
        self.mock_client.insert.assert_called_once()
        # Check that the right collection was used
        self.assertEqual(self.mock_client.insert.call_args[0][0], self.test_collection)
        # The DataPacket passed to insert should have the same record_id as our model
        self.assertEqual(self.mock_client.insert.call_args[0][1].record_id, test_model.record_id)

    def test_search_route(self):
        """Test the search route."""
        # Setup test data
        test_query_model = self.create_test_query_packet_model()
        test_data_packet = self.create_test_data_packet()

        # Create a real QueryPacket using the real DataModelUtils
        expected_query_packet = DataModelUtils.convert_to_query_packet(test_query_model)

        # Setup search result
        search_result = SearchDataPacket(data_packet=test_data_packet, distance=0.5)
        self.mock_client.search.return_value = [search_result]

        # Create a SearchRequest instance instead of using a raw dict
        search_request = SearchRequest(query=test_query_model)

        # Make the request with proper serialization
        response = self.client.post(
            f"/collections/{self.test_collection}/search",
            json=search_request.dict(by_alias=True, exclude_none=True)
        )

        # Check response
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["distance"], 0.5)
        self.assertEqual(results[0]["data_packet"]["record_id"], self.test_record_id)

        # Verify method calls
        self.mock_client.search.assert_called_once()
        # Check that the right collection was used
        self.assertEqual(self.mock_client.search.call_args[0][0], self.test_collection)
        # The QueryPacket should have the same attributes as our converted model
        query_packet_arg = self.mock_client.search.call_args[0][1]
        self.assertEqual(query_packet_arg.k, expected_query_packet.k)
        self.assertEqual(query_packet_arg.where, expected_query_packet.where)
        self.assertEqual(query_packet_arg.where_document, expected_query_packet.where_document)
        np.testing.assert_array_equal(query_packet_arg.query_vector, expected_query_packet.query_vector)

    def test_get_record_route(self):
        """Test the get_record route."""
        # Setup test data
        test_packet = self.create_test_data_packet()
        self.mock_client.get.return_value = test_packet

        # Make the request
        response = self.client.get(
            f"/collections/{self.test_collection}/records/{self.test_record_id}"
        )

        # Check response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["record_id"], self.test_record_id)

        # Verify method calls
        self.mock_client.get.assert_called_once_with(self.test_collection, self.test_record_id)

    def test_delete_record_route(self):
        """Test the delete_record route."""
        # Setup test data
        test_packet = self.create_test_data_packet()
        self.mock_client.delete.return_value = test_packet

        # Make the request
        response = self.client.delete(
            f"/collections/{self.test_collection}/records/{self.test_record_id}"
        )

        # Check response
        self.assertEqual(200, response.status_code)
        result = response.json()
        self.assertEqual(self.test_record_id, result["record_id"])

        # Verify method calls
        self.mock_client.delete.assert_called_once_with(self.test_collection, self.test_record_id)

    def test_error_handling_checksum_validation_failure(self):
        """Test error handling for checksum validation failure."""
        # Create a test model with invalid data that will fail checksum validation
        test_model = self.create_test_data_packet_model()

        # Patch the DataModelUtils.convert_to_data_packet to simulate validation failure
        with patch.object(DataModelUtils, 'convert_to_data_packet') as mock_convert:
            mock_convert.side_effect = ChecksumValidationFailureError("Invalid checksum")

            # Create an InsertRequest and properly serialize it
            insert_request = InsertRequest(packet=test_model)

            # Make the request with proper serialization
            response = self.client.post(
                f"/collections/{self.test_collection}/insert",
                json=insert_request.dict(by_alias=True, exclude_none=True)
            )

            # Check response
            self.assertEqual(response.status_code, 400)
            result = response.json()
            self.assertIn("Checksum validation failed", result["detail"])

    def test_error_handling_record_not_found(self):
        """Test error handling for record isn't found."""
        # Setup mocks to raise RecordNotFoundError
        self.mock_client.get.side_effect = RecordNotFoundError(f"Record {self.test_record_id} not found")

        # Make the request
        response = self.client.get(
            f"/collections/{self.test_collection}/records/{self.test_record_id}"
        )

        # Check response
        self.assertEqual(response.status_code, 404)
        result = response.json()
        self.assertIn("Record not found", result["detail"])

    def test_error_handling_generic_exception(self):
        """Test error handling for generic exceptions."""
        # Setup mocks to raise a generic exception
        self.mock_client.get.side_effect = Exception("Database connection error")

        # Make the request
        response = self.client.get(
            f"/collections/{self.test_collection}/records/{self.test_record_id}"
        )

        # Check response
        self.assertEqual(response.status_code, 400)
        result = response.json()
        self.assertEqual(result["detail"], "Database connection error")

    def test_empty_search_results(self):
        """Test search with empty results."""
        # Setup test data
        test_query_model = self.create_test_query_packet_model()

        # Setup mocks for empty results
        self.mock_client.search.return_value = []

        # Create a SearchRequest instance and properly serialize it
        search_request = SearchRequest(query=test_query_model)

        # Make the request with proper serialization
        response = self.client.post(
            f"/collections/{self.test_collection}/search",
            json=search_request.dict(by_alias=True, exclude_none=True)
        )

        # Check response
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), 0)

        # Verify method calls
        self.mock_client.search.assert_called_once()

    def test_round_trip_data_packet_conversion(self):
        """Test that data packets correctly round-trip through the API."""
        # Create a test model
        original_model = self.create_test_data_packet_model()

        # Create a real DataPacket
        packet = DataModelUtils.convert_to_data_packet(original_model)
        self.mock_client.insert.return_value = packet

        # Create an InsertRequest instance and properly serialize it
        insert_request = InsertRequest(packet=original_model)

        # Make the insert request with proper serialization
        response = self.client.post(
            f"/collections/{self.test_collection}/insert",
            json=insert_request.dict(by_alias=True, exclude_none=True)
        )

        # Check response
        self.assertEqual(response.status_code, 200)
        result = response.json()

        # Create a model from the response and compare with the original
        result_model = DataPacketModel(**result)

        # Check core fields match
        self.assertEqual(result_model.record_id, original_model.record_id)
        self.assertEqual(result_model.original_data, original_model.original_data)
        self.assertEqual(result_model.metadata, original_model.metadata)

        # Compare vectors
        original_vector = original_model.vector.to_numpy()
        result_vector = result_model.vector.to_numpy()
        np.testing.assert_array_almost_equal(original_vector, result_vector)

    def test_healthz_endpoint(self):
        """Liveness probe should return status ok."""
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_readyz_endpoint(self):
        """Readiness probe should return status ready."""
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})

    def test_metrics_endpoint(self):
        """Test the metrics endpoint returns Prometheus metrics correctly."""
        # First, make some API calls to generate metrics
        # Insert operation to increment counter
        test_model = self.create_test_data_packet_model()
        test_packet = DataModelUtils.convert_to_data_packet(test_model)
        self.mock_client.insert.return_value = test_packet
        insert_request = InsertRequest(packet=test_model)

        self.client.post(
            f"/collections/{self.test_collection}/insert",
            json=insert_request.dict(by_alias=True, exclude_none=True)
        )

        # Now call the metrics endpoint
        response = self.client.get("/metrics")

        # Check response status code
        self.assertEqual(response.status_code, 200)

        # Check content type
        self.assertEqual(response.headers["content-type"], CONTENT_TYPE_LATEST)

        # Check that the response contains expected metrics
        content = response.content.decode('utf-8')

        # Verify counter-metrics are present
        self.assertIn('vecraft_insert_total', content)

        # Verify that insert counter was incremented
        self.assertIn('vecraft_insert_total 1.0', content)

        # Verify other metrics exist
        self.assertIn('vecraft_search_total', content)
        self.assertIn('vecraft_get_total', content)
        self.assertIn('vecraft_delete_total', content)

        # Verify histogram metrics exist
        self.assertIn('vecraft_insert_latency_bucket', content)
        self.assertIn('vecraft_search_latency_bucket', content)
        self.assertIn('vecraft_get_latency_bucket', content)
        self.assertIn('vecraft_delete_latency_bucket', content)

    def test_create_collection_route(self):
        """Test creating a new collection."""
        # Mock create_collection 方法，避免实际写入
        self.mock_client.create_collection.return_value.to_dict.return_value = {
            "name": self.test_collection,
            "dim": 5,
            "vector_type": "float",
            "checksum_algorithm": "sha256",
            "checksum": "dummy"
        }

        payload = {
            "dim": 5,
            "vector_type": "float",
            "checksum_algorithm": "sha256"
        }

        response = self.client.post(
            f"/collections/{self.test_collection}/create",
            json=payload
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "created")
        self.assertEqual(response.json()["collection"]["name"], self.test_collection)

        self.mock_client.create_collection.assert_called_once()

    def test_list_collections_route(self):
        """Test listing all collections."""
        mock_schema = MagicMock()
        mock_schema.to_dict.return_value = {
            "name": self.test_collection,
            "dim": 5,
            "vector_type": "float",
            "checksum_algorithm": "sha256",
            "checksum": "dummy"
        }
        self.mock_client.list_collections.return_value = [mock_schema]

        response = self.client.get("/collections")

        self.assertEqual(response.status_code, 200)
        collections = response.json()
        self.assertEqual(len(collections), 1)
        self.assertEqual(collections[0]["name"], self.test_collection)

        self.mock_client.list_collections.assert_called_once()


if __name__ == '__main__':
    unittest.main()