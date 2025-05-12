import asyncio
import os
import stat
from contextlib import asynccontextmanager
from functools import wraps
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge

from src.vecraft_api.rest.data_model_utils import DataModelUtils
from src.vecraft_api.rest.data_models import DataPacketModel, InsertRequest, SearchRequest
from src.vecraft_db.core.data_model.exception import ChecksumValidationFailureError, RecordNotFoundError


class VecraftRestAPI:
    def __init__(self, root: str):
        # Ensure the root directory exists with safe permissions (owner-only)
        if os.path.exists(root):
            # Remove group and other write permissions
            current_mode = os.stat(root).st_mode
            safe_mode = current_mode & ~stat.S_IWGRP & ~stat.S_IWOTH
            os.chmod(root, safe_mode)
        else:
            os.makedirs(root, mode=0o700, exist_ok=True)

        # Import inside init so tests can patch K8sAwareVecraftClient
        from src.vecraft_db.client.k8s_aware_vecraft_client import K8sAwareVecraftClient
        self.client = K8sAwareVecraftClient(root)

        @asynccontextmanager
        async def lifespan():
            # ==== Startup ====
            yield
            # ==== Shutdown ====
            await asyncio.to_thread(self.client.close)

        self.app = FastAPI(
            title="Vecraft Vector Database API",
            description="RESTful API for Vecraft vector database operations",
            version="1.0.0",
            lifespan=lifespan
        )

        # Liveness probe: indicates the app is up
        @self.app.get("/healthz", include_in_schema=False)
        async def healthz():
            return JSONResponse({"status": "ok"})

        # Readiness probe: indicates the app is ready to serve
        @self.app.get("/readyz", include_in_schema=False)
        async def readyz():
            # TODO implement deeper checks # NOSONAR
            try:
                return JSONResponse({"status": "ready"})
            except Exception:
                raise HTTPException(status_code=503, detail="not ready")

        self.search_counter = Counter('vecraft_search_total', 'Total number of search operations')
        self.insert_counter = Counter('vecraft_insert_total', 'Total number of insert operations')
        self.get_counter = Counter('vecraft_get_total', 'Total number of get operations')
        self.delete_counter = Counter('vecraft_delete_total', 'Total number of delete operations')
        self.search_latency = Histogram('vecraft_search_latency', 'Search operation latency')
        self.insert_latency = Histogram('vecraft_insert_latency', 'Insert operation latency')
        self.get_latency = Histogram('vecraft_get_latency', 'Get operation latency')
        self.delete_latency = Histogram('vecraft_delete_latency', 'Delete operation latency')
        self.collection_size = Gauge('vecraft_collection_size', 'Number of records in collection', ['collection'])

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
            # increase counter
            self.insert_counter.inc()

            with self.insert_latency.time():
                packet = DataModelUtils.convert_to_data_packet(request.packet)
                preimage = self.client.insert(collection, packet)
                return DataModelUtils.convert_from_data_packet(preimage)

        @self.app.post("/collections/{collection}/search", response_model=List[Dict[str, Any]])
        @with_error_handling()
        async def search(collection: str, request: SearchRequest):
            # increase counter
            self.search_counter.inc()

            with self.search_latency.time():
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
            # increase counter
            self.get_counter.inc()

            with self.get_latency.time():
                result = self.client.get(collection, record_id)
                return DataModelUtils.convert_from_data_packet(result)

        @self.app.delete("/collections/{collection}/records/{record_id}", response_model=DataPacketModel)
        @with_error_handling()
        async def delete_record(collection: str, record_id: str):
            # increase counter
            self.delete_counter.inc()

            with self.delete_latency.time():
                preimage = self.client.delete(collection, record_id)
                return DataModelUtils.convert_from_data_packet(preimage)

        @self.app.get("/metrics")
        async def metrics():
            # TODO Get metrics for collection sizes # NOSONAR

            data = await asyncio.to_thread(generate_latest)
            return Response(content=data, media_type=CONTENT_TYPE_LATEST)