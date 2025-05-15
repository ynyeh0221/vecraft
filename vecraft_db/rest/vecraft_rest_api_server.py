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

from vecraft_data_model.data_model_utils import DataModelUtils
from vecraft_data_model.data_models import InsertRequest, DataPacketModel, SearchRequest, CreateCollectionRequest
from vecraft_data_model.index_packets import CollectionSchema
from vecraft_exception_model.exception import RecordNotFoundError, ChecksumValidationFailureError

class VecraftRestAPI:
    """
    VecraftRestAPI: HTTP Interface for Vecraft Vector Database

    This REST API provides vector database functionality including collection management,
    record insertion, retrieval, deletion, and vector-based search.

    Base URL:
        http://127.0.0.1:8000

    ---

    Health & Readiness
    ------------------

    1. Health Check:
        Check if the server is running.

        curl http://127.0.0.1:8000/healthz

    2. Readiness Check:
        Check if the server is ready to serve requests.

        curl http://127.0.0.1:8000/readyz

    ---

    Collection Management
    ---------------------

    1. Create a Collection:
        POST /collections/{collection}/create

        Request JSON:
        {
            "dim": 128,
            "vector_type": "float",
            "checksum_algorithm": "sha256"
        }

        Example:
        curl -X POST http://127.0.0.1:8000/collections/my_collection/create \
             -H "Content-Type: application/json" \
             -d '{"dim": 128, "vector_type": "float", "checksum_algorithm": "sha256"}'

    2. List Collections:
        GET /collections

        Example:
        curl http://127.0.0.1:8000/collections

    ---

    Insert Records
    --------------

    POST /collections/{collection}/insert

    Request JSON format:
    {
        "packet": {
            "record_id": "rec-001",
            "type": "RECORD",
            "vector": {
                "b64": "<base64-encoded-bytes>",
                "dtype": "float64",
                "shape": [128]
            },
            "original_data": null,
            "metadata": {
                "tag": "demo"
            },
            "checksum_algorithm": "sha256"
        }
    }

    Example:
    curl -X POST http://127.0.0.1:8000/collections/my_collection/insert \
         -H "Content-Type: application/json" \
         -d @insert_payload.json

    ---

    Search Records
    --------------

    POST /collections/{collection}/search

    Request JSON format:
    {
        "query": {
            "query_vector": {
                "b64": "<base64-encoded-bytes>",
                "dtype": "float64",
                "shape": [128]
            },
            "k": 10,
            "where": {
                "tag": "demo"
            },
            "where_document": null,
            "checksum_algorithm": "sha256"
        }
    }

    Example:
    curl -X POST http://127.0.0.1:8000/collections/my_collection/search \
         -H "Content-Type: application/json" \
         -d @search_payload.json

    ---

    Get Record
    ----------

    GET /collections/{collection}/records/{record_id}

    Example:
    curl http://127.0.0.1:8000/collections/my_collection/records/rec-001

    ---

    Delete Record
    -------------

    DELETE /collections/{collection}/records/{record_id}

    Example:
    curl -X DELETE http://127.0.0.1:8000/collections/my_collection/records/rec-001

    ---

    Metrics (Prometheus)
    ---------------------

    GET /metrics

    Returns Prometheus metrics for monitoring.

    Example:
    curl http://127.0.0.1:8000/metrics

    ---
    """
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
        from vecraft_db.client.k8s_aware_vecraft_client import K8sAwareVecraftClient
        self.client = K8sAwareVecraftClient(root)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
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

        @self.app.post("/collections/{collection}/create")
        @with_error_handling()
        async def create_collection(collection: str, request: CreateCollectionRequest):
            try:
                collection_schema = CollectionSchema(
                    name=collection,
                    dim=request.dim,
                    vector_type=request.vector_type,
                    checksum_algorithm=request.checksum_algorithm
                )

                self.client.create_collection(collection_schema)

                return {
                    "status": "created",
                    "collection": collection_schema.to_dict()
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Collection creation failed: {str(e)}")

        @self.app.get("/collections")
        @with_error_handling()
        async def list_collections():
            collections = self.client.list_collections()
            return [c.to_dict() for c in collections]

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

app = VecraftRestAPI(root="./vecraft-data").app