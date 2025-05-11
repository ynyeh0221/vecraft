import inspect
from typing import List

from src.vecraft_api.data_model_utils import DataModelUtils
from src.vecraft_api.data_models import DataPacketModel
from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.query_packet import QueryPacket
from src.vecraft_db.core.data_model.search_data_packet import SearchDataPacket


class VecraftFastAPIClient:
    """Client for interacting with Vecraft FastAPI server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        import httpx
        self.session = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()


    async def insert(
        self,
        collection: str,
        data_packet: DataPacket
    ) -> DataPacket:
        """
        Insert a record

        Args:
            collection: Collection name
        """
        data_model = DataModelUtils.convert_from_data_packet(data_packet)
        response = await self.session.post(
            f"{self.base_url}/collections/{collection}/insert",
            json={"packet": data_model}
        )
        # Handle both sync and async raise_for_status
        rf = response.raise_for_status()
        if inspect.isawaitable(rf):
            await rf
        # Handle both sync and async json()
        json_result = response.json()
        if inspect.isawaitable(json_result):
            json_result = await json_result
        return DataModelUtils.convert_to_data_packet(json_result)

    async def search(
        self,
        collection: str,
        query_packet: QueryPacket
    ) -> List[SearchDataPacket]:
        """
        Search record(s) similar to target vector

        Args:
            collection: Collection name
        """
        query_model = DataModelUtils.convert_from_query_packet(query_packet)
        response = await self.session.post(
            f"{self.base_url}/collections/{collection}/search",
            json={"query": query_model}
        )
        # Handle raise_for_status
        rf = response.raise_for_status()
        if inspect.isawaitable(rf):
            await rf
        items_raw = response.json()
        if inspect.isawaitable(items_raw):
            items = await items_raw
        else:
            items = items_raw
        results: List[SearchDataPacket] = []
        for item in items:
            dp_model = DataPacketModel(**item['data_packet'])
            dp = DataModelUtils.convert_to_data_packet(dp_model)
            distance = item['distance']
            results.append(SearchDataPacket(data_packet=dp, distance=distance))
        return results

    async def get(
        self,
        collection: str,
        record_id: str
    ) -> DataPacket:
        """
        Fetch a single record by ID
        """
        response = await self.session.get(
            f"{self.base_url}/collections/{collection}/records/{record_id}"
        )
        rf = response.raise_for_status()
        if inspect.isawaitable(rf):
            await rf
        json_result = response.json()
        if inspect.isawaitable(json_result):
            json_result = await json_result
        return DataModelUtils.convert_to_data_packet(json_result)

    async def delete(
        self,
        collection: str,
        record_id: str
    ) -> DataPacket:
        """
        Delete a record by ID
        """
        response = await self.session.delete(
            f"{self.base_url}/collections/{collection}/records/{record_id}"
        )
        rf = response.raise_for_status()
        if inspect.isawaitable(rf):
            await rf
        json_result = response.json()
        if inspect.isawaitable(json_result):
            json_result = await json_result
        return DataModelUtils.convert_to_data_packet(json_result)