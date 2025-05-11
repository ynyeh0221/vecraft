from typing import Dict, Any, List, Optional
from functools import wraps
import os
import stat
from fastapi import FastAPI, HTTPException

from src.vecraft_api.rest.data_model_utils import DataModelUtils
from src.vecraft_api.rest.data_models import DataPacketModel, InsertRequest, SearchRequest
from src.vecraft_db.core.data_model.exception import ChecksumValidationFailureError, RecordNotFoundError


class VecraftAPI:
    def __init__(self, root: str, vector_index_params: Optional[Dict[str, Any]] = None):
        # Ensure the root directory exists with safe permissions (owner-only)
        if os.path.exists(root):
            # Remove group and other write permissions
            current_mode = os.stat(root).st_mode
            safe_mode = current_mode & ~stat.S_IWGRP & ~stat.S_IWOTH
            os.chmod(root, safe_mode)
        else:
            os.makedirs(root, mode=0o700, exist_ok=True)

        # Import inside init so tests can patch VecraftClient
        from src.vecraft_db.client.vecraft_client import VecraftClient
        self.client = VecraftClient(root, vector_index_params)

        self.app = FastAPI(
            title="Vecraft Vector Database API",
            description="RESTful API for Vecraft vector database operations",
            version="1.0.0"
        )
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes with mandatory checksum validation"""

        def with_error_handling():
            """
            Decorator factory to wrap route handlers in shared exception logic.
            Not-found errors become 404, other exceptions become 400.
            """

            def decorator(func):
                @wraps(func)
                async def wrapper(*args, **kwargs):
                    try:
                        return await func(*args, **kwargs)
                    # TODO map all customer-facing exceptions # NOSONAR
                    except ChecksumValidationFailureError as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Checksum validation failed: {e}"
                        )
                    except RecordNotFoundError as e:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Record not found: {e}"
                        )
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=str(e))
                return wrapper
            return decorator

        @self.app.post("/collections/{collection}/insert", response_model=DataPacketModel)
        @with_error_handling()
        async def insert(collection: str, request: InsertRequest):
            packet = DataModelUtils.convert_to_data_packet(request.packet)
            preimage = self.client.insert(collection, packet)
            return DataModelUtils.convert_from_data_packet(preimage)

        @self.app.post("/collections/{collection}/search", response_model=List[Dict[str, Any]])
        @with_error_handling()
        async def search(collection: str, request: SearchRequest):
            query_packet = DataModelUtils.convert_to_query_packet(request.query)
            results = self.client.search(collection, query_packet)
            return [
                {
                    "data_packet": DataModelUtils.convert_from_data_packet(r.data_packet).dict(),
                    "distance": r.distance
                }
                for r in results
            ]

        @self.app.get("/collections/{collection}/records/{record_id}", response_model=DataPacketModel)
        @with_error_handling()
        async def get_record(collection: str, record_id: str):
            result = self.client.get(collection, record_id)
            return DataModelUtils.convert_from_data_packet(result)

        @self.app.delete("/collections/{collection}/records/{record_id}", response_model=DataPacketModel)
        @with_error_handling()
        async def delete_record(collection: str, record_id: str):
            preimage = self.client.delete(collection, record_id)
            return DataModelUtils.convert_from_data_packet(preimage)