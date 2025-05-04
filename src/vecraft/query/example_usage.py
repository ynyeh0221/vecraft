import os
import shutil
from pathlib import Path

import numpy as np
from src.vecraft.core.storage_interface import StorageEngine
from src.vecraft.engine.vector_db import VectorDB
from src.vecraft.index.record_location.json_based_location_index import JsonRecordLocationIndex
from src.vecraft.index.record_location.location_index_interface import RecordLocationIndex
from src.vecraft.index.record_vector.hnsw import HNSW
from src.vecraft.metadata.catalog import JsonCatalog
from src.vecraft.query.executor import Executor
from src.vecraft.query.planner import Planner
from src.vecraft.storage.file_mmap import MMapStorage


def setup_db(storage_path: str, catalog_path: str, location_path: str):
    storage: StorageEngine = MMapStorage(storage_path)
    catalog = JsonCatalog(catalog_path)
    location_index: RecordLocationIndex = JsonRecordLocationIndex(Path(location_path))

    def vector_index(kind: str, dim: int):
        if kind == "hnsw":
            return HNSW(dim=dim, M=16, ef_construction=200)
        raise ValueError(f"Unknown index kind: {kind}")

    db = VectorDB(storage=storage, catalog=catalog,
                  vector_index=vector_index, location_index=location_index)
    return db, catalog


if __name__ == "__main__":
    # Paths for storage, catalog, and location index
    storage_path = "./test/storage.json"
    catalog_path = "./test/catalog.json"
    location_path = "./test/location.json"

    # Initialize database
    db, catalog = setup_db(storage_path, catalog_path, location_path)
    planner = Planner()
    executor = Executor(db)

    # Ensure collection exists
    collection_name = "test_collection"
    if collection_name not in catalog.list_collections():
        catalog.create_collection(collection_name, dim=128, vector_type="float32")

    # Prepare multiple records to insert
    records = [
        {"text": "Hello, world!", "tags": ["greeting"]},
        {"text": "Goodbye, world!", "tags": ["farewell"]},
        {"text": "The quick brown fox", "tags": ["animal", "phrase"]},
        {"text": "Lorem ipsum dolor sit amet", "tags": ["lorem"]},
    ]

    # Insert multiple records
    inserted_ids = []
    for data in records:
        # Create random vector for each record
        dim = catalog.get_schema(collection_name).field.dim
        vector = np.random.rand(dim).astype(np.float32)

        # Plan and execute insert
        plan = planner.plan_insert(
            collection=collection_name,
            original_data=data,
            vector=vector,
            metadata={"tags": data.get("tags", [])}
        )
        rec_id = executor.execute(plan)
        inserted_ids.append(rec_id)
        print(f"Inserted record {rec_id} -> {data['text']}")

    # Plan & execute a search across tags 'greeting' and 'farewell'
    # Combine tags filter to search for records with either tag
    query_vec = np.random.rand(catalog.get_schema(collection_name).field.dim).astype(np.float32)
    search_plan = planner.plan_search(
        collection=collection_name,
        query_vector=query_vec,
        k=3,
        where={"tags": ["greeting", "farewell"]}
    )
    search_results = executor.execute(search_plan)
    print("\nSearch results for tags ['greeting','farewell']:")
    for rank, res in enumerate(search_results, start=1):
        print(f" {rank}. ID={res['id']}, distance={res['distance']}, data={res['original_data']}")

    # Plan & execute get for all inserted IDs
    print("\nFetching all inserted records:")
    for rec_id in inserted_ids:
        get_plan = planner.plan_get(collection_name, rec_id)
        record = executor.execute(get_plan)
        print(f"Fetched ID={rec_id}: {record}")

        # Cleanup test files and directory
        test_dir = Path(storage_path).parent
        for path in [storage_path, catalog_path, location_path]:
            try:
                os.remove(path)
                print(f"Removed file {path}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error removing {path}: {e}")
        try:
            shutil.rmtree(test_dir)
            print(f"Removed directory {test_dir}")
        except Exception:
            pass
