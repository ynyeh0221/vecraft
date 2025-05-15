import unittest
from unittest.mock import patch, AsyncMock

import httpx
import numpy as np

from vecraft_data_model.data_model_utils import DataModelUtils
from vecraft_data_model.data_models import DataPacketModel, NumpyArray, QueryPacketModel, CreateCollectionRequest
from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.query_packet import QueryPacket
from vecraft_data_model.search_data_packet import SearchDataPacket
from vecraft_rest_client.vecraft_rest_api_client import VecraftFastAPIClient


class TestVecraftFastAPIClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self): #NOSONAR
        """Set up test fixtures before each test method."""
        # Setup test vectors
        rng = np.random.default_rng(42)  # Use seeded generator for reproducibility
        self.test_vector = rng.random(5)
        self.test_query_vector = rng.random(5)

        # Create test data
        self.test_collection = "test_collection"
        self.test_record_id = "record_123"
        self.test_original_data = {"content": "Test content", "id": 123}
        self.test_metadata = {"source": "test", "timestamp": "2024-01-01"}
        self.test_checksum_algorithm = "sha256"

        # Create a test DataPacket
        self.data_packet = DataPacket.create_record(
            record_id=self.test_record_id,
            original_data=self.test_original_data,
            metadata=self.test_metadata,
            checksum_algorithm=self.test_checksum_algorithm,
            vector=self.test_vector
        )

        # Create a test DataPacketModel
        self.data_packet_model = DataPacketModel(
            type="RECORD",
            record_id=self.test_record_id,
            original_data=self.test_original_data,
            metadata=self.test_metadata,
            checksum_algorithm=self.test_checksum_algorithm,
            vector=NumpyArray.from_numpy(self.test_vector),
            checksum=self.data_packet.checksum
        )

        # Create a test QueryPacket
        self.query_packet = QueryPacket(
            query_vector=self.test_query_vector,
            k=10,
            where={"category": "test"},
            where_document={"content": "test"},
            checksum_algorithm=self.test_checksum_algorithm
        )

        # Create a test QueryPacketModel
        self.query_packet_model = QueryPacketModel(
            query_vector=NumpyArray.from_numpy(self.test_query_vector),
            k=10,
            where={"category": "test"},
            where_document={"content": "test"},
            checksum_algorithm=self.test_checksum_algorithm,
            checksum=self.query_packet.checksum
        )

        # Create the client
        self.client = VecraftFastAPIClient(base_url="https://testserver")

        # Mock the httpx.AsyncClient
        self.mock_session = AsyncMock()
        self.client.session = self.mock_session

    async def asyncTearDown(self): #NOSONAR
        """Clean up after each test method."""
        if self.client and self.client.session:
            # Ensure the session is closed
            await self.client.session.aclose()

    async def test_context_manager(self):
        """Test the async context manager protocol."""
        # Mock httpx.AsyncClient
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Use client as context manager
            async with VecraftFastAPIClient(base_url="https://testserver") as client:
                self.assertIsNotNone(client.session)
                self.assertEqual(client.base_url, "https://testserver")

            # Verify session was closed
            mock_client.aclose.assert_called_once()

    async def test_create_collection(self):
        """Test the create_collection method."""
        # Mock the session post method
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        expected = {"status": "created", "collection": {"name": self.test_collection, "dim": 5, "vector_type": "float",
                                                        "checksum_algorithm": "sha256", "checksum": "abc123"}}
        mock_response.json.return_value = expected
        self.client.session.post.return_value = mock_response

        # Call method
        result = await self.client.create_collection(
            collection=self.test_collection,
            dim=5,
            vector_type="float",
            checksum_algorithm="sha256"
        )

        self.assertEqual(result, expected)
        self.client.session.post.assert_called_once_with(
            f"https://testserver/collections/{self.test_collection}/create",
            json=CreateCollectionRequest(dim=5, vector_type="float", checksum_algorithm="sha256").dict()
        )

    async def test_list_collections(self):
        """Test the list_collections method."""
        # Mock the session get method
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        expected = [
            {"name": "col1", "dim": 3, "vector_type": "float", "checksum_algorithm": "sha256", "checksum": "xyz"},
            {"name": "col2", "dim": 5, "vector_type": "binary", "checksum_algorithm": "sha256", "checksum": "def"}
        ]
        mock_response.json.return_value = expected
        self.client.session.get.return_value = mock_response

        # Call method
        result = await self.client.list_collections()

        self.assertEqual(result, expected)
        self.client.session.get.assert_called_once_with(
            "https://testserver/collections"
        )

    async def test_insert(self):
        """Test the insert method."""
        # Mock the session post method
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        # FIX: Return the value directly, not a coroutine
        mock_response.json.return_value = self.data_packet_model.dict()
        self.client.session.post.return_value = mock_response

        # Patch the DataModelUtils methods
        with patch.object(DataModelUtils, 'convert_from_data_packet') as mock_convert_from, \
                patch.object(DataModelUtils, 'convert_to_data_packet') as mock_convert_to:
            # Setup mock returns
            mock_convert_from.return_value = self.data_packet_model
            mock_convert_to.return_value = self.data_packet

            # Call the method
            result = await self.client.insert(
                self.test_collection,
                self.data_packet
            )

            # Verify the result
            self.assertEqual(result, self.data_packet)

            # Verify mocks were called correctly
            mock_convert_from.assert_called_once_with(self.data_packet)
            mock_convert_to.assert_called_once()
            self.client.session.post.assert_called_once_with(
                f"https://testserver/collections/{self.test_collection}/insert",
                json={"packet": self.data_packet_model}
            )

    async def test_search(self):
        """Test the search method."""
        # Create a mock search result
        search_result = {
            "data_packet": self.data_packet_model.dict(),
            "distance": 0.5
        }

        # Mock the session post method
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        # FIX: Return the value directly, not a coroutine
        mock_response.json.return_value = [search_result]
        self.client.session.post.return_value = mock_response

        # Patch the DataModelUtils methods
        with patch.object(DataModelUtils, 'convert_from_query_packet') as mock_convert_from, \
                patch.object(DataModelUtils, 'convert_to_data_packet') as mock_convert_to:
            # Setup mock returns
            mock_convert_from.return_value = self.query_packet_model
            mock_convert_to.return_value = self.data_packet

            # Call the method
            results = await self.client.search(
                self.test_collection,
                self.query_packet
            )

            # Verify the result
            self.assertEqual(len(results), 1)
            self.assertIsInstance(results[0], SearchDataPacket)
            self.assertEqual(results[0].data_packet, self.data_packet)
            self.assertEqual(results[0].distance, 0.5)

            # Verify mocks were called correctly
            mock_convert_from.assert_called_once_with(self.query_packet)
            mock_convert_to.assert_called_once()
            self.client.session.post.assert_called_once_with(
                f"https://testserver/collections/{self.test_collection}/search",
                json={"query": self.query_packet_model}
            )

    async def test_get(self):
        """Test the get method."""
        # Mock the session get method
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        # FIX: Return the value directly, not a coroutine
        mock_response.json.return_value = self.data_packet_model.dict()
        self.client.session.get.return_value = mock_response

        # Patch the DataModelUtils.convert_to_data_packet method
        with patch.object(DataModelUtils, 'convert_to_data_packet') as mock_convert:
            mock_convert.return_value = self.data_packet

            # Call the method
            result = await self.client.get(
                self.test_collection,
                self.test_record_id
            )

            # Verify the result
            self.assertEqual(result, self.data_packet)

            # Verify mocks were called correctly
            mock_convert.assert_called_once()
            self.client.session.get.assert_called_once_with(
                f"https://testserver/collections/{self.test_collection}/records/{self.test_record_id}"
            )

    async def test_delete(self):
        """Test the delete method."""
        # Mock the session delete method
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        # FIX: Return the value directly, not a coroutine
        mock_response.json.return_value = self.data_packet_model.dict()
        self.client.session.delete.return_value = mock_response

        # Patch the DataModelUtils.convert_to_data_packet method
        with patch.object(DataModelUtils, 'convert_to_data_packet') as mock_convert:
            mock_convert.return_value = self.data_packet

            # Call the method
            result = await self.client.delete(
                self.test_collection,
                self.test_record_id
            )

            # Verify the result
            self.assertEqual(result, self.data_packet)

            # Verify mocks were called correctly
            mock_convert.assert_called_once()
            self.client.session.delete.assert_called_once_with(
                f"https://testserver/collections/{self.test_collection}/records/{self.test_record_id}"
            )

    async def test_http_error_handling(self):
        """Test handling of HTTP errors."""
        # Mock the session post method to raise an error
        mock_response = AsyncMock()
        # FIX: Make raise_for_status throw the exception directly
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=httpx.Request("POST", "https://testserver"),
            response=httpx.Response(404)
        )
        # We don't need json() to return anything since raise_for_status will throw
        self.client.session.post.return_value = mock_response

        # Patch the DataModelUtils.convert_from_data_packet method
        with patch.object(DataModelUtils, 'convert_from_data_packet') as mock_convert:
            mock_convert.return_value = self.data_packet_model

            # Call the method and expect an exception
            with self.assertRaises(httpx.HTTPStatusError):
                await self.client.insert(
                    self.test_collection,
                    self.data_packet
                )

    async def test_empty_search_results(self):
        """Test handling of empty search results."""
        # Mock the session post method
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        # FIX: Return the value directly, not a coroutine
        mock_response.json.return_value = []  # Empty results
        self.client.session.post.return_value = mock_response

        # Patch the DataModelUtils.convert_from_query_packet method
        with patch.object(DataModelUtils, 'convert_from_query_packet') as mock_convert:
            mock_convert.return_value = self.query_packet_model

            # Call the method
            results = await self.client.search(
                self.test_collection,
                self.query_packet
            )

            # Verify the result is an empty list
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 0)

    async def test_client_with_custom_base_url(self):
        """Test creating client with custom base URL."""
        custom_url = "https://custom-server:9000"
        client = VecraftFastAPIClient(base_url=custom_url)
        self.assertEqual(client.base_url, custom_url)

    async def test_search_with_different_parameters(self):
        """Test search with different parameters."""
        # Create a test query packet with different parameters
        different_query_packet = QueryPacket(
            query_vector=self.test_query_vector,
            k=5,  # Different k value
            where={"status": "active"},  # Different where condition
            where_document={"language": "en"},  # Different where_document
            checksum_algorithm=self.test_checksum_algorithm
        )

        # Convert to model
        different_query_model = DataModelUtils.convert_from_query_packet(different_query_packet)

        # Mock the session post method
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        # FIX: Return the value directly, not a coroutine
        mock_response.json.return_value = []  # No results
        self.client.session.post.return_value = mock_response

        # Patch the DataModelUtils methods
        with patch.object(DataModelUtils, 'convert_from_query_packet') as mock_convert_from:
            mock_convert_from.return_value = different_query_model

            # Call the method
            await self.client.search(
                self.test_collection,
                different_query_packet
            )

            # Verify the correct query was sent
            self.client.session.post.assert_called_once_with(
                f"https://testserver/collections/{self.test_collection}/search",
                json={"query": different_query_model}
            )

    async def test_deserialize_complex_search_results(self):
        """Test deserialization of complex search results."""
        # Create multiple search results with different distances
        search_results = [
            {
                "data_packet": self.data_packet_model.dict(),
                "distance": 0.1
            },
            {
                "data_packet": self.data_packet_model.dict(),
                "distance": 0.2
            },
            {
                "data_packet": self.data_packet_model.dict(),
                "distance": 0.3
            }
        ]

        # Mock the session post method
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        # FIX: Return the value directly, not a coroutine
        mock_response.json.return_value = search_results
        self.client.session.post.return_value = mock_response

        # Patch the DataModelUtils methods
        with patch.object(DataModelUtils, 'convert_from_query_packet') as mock_convert_from, \
                patch.object(DataModelUtils, 'convert_to_data_packet') as mock_convert_to:
            # Setup mock returns
            mock_convert_from.return_value = self.query_packet_model
            mock_convert_to.return_value = self.data_packet

            # Call the method
            results = await self.client.search(
                self.test_collection,
                self.query_packet
            )

            # Verify we got the expected number of results
            self.assertEqual(len(results), 3)

            # Verify the distances were preserved
            self.assertEqual(results[0].distance, 0.1)
            self.assertEqual(results[1].distance, 0.2)
            self.assertEqual(results[2].distance, 0.3)

            # Verify convert_to_data_packet was called 3 times
            self.assertEqual(mock_convert_to.call_count, 3)


if __name__ == '__main__':
    unittest.main()