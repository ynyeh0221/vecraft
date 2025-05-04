import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

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

@pytest.fixture
def temp_paths(tmp_path):
    test_dir = tmp_path / "test_db"
    test_dir.mkdir()
    storage_path = str(test_dir / "storage.json")
    catalog_path = str(test_dir / "catalog.json")
    location_path = str(test_dir / "location.json")
    yield storage_path, catalog_path, location_path
    shutil.rmtree(test_dir)


def test_insert_search_and_fetch_consistency(temp_paths):
    storage_path, catalog_path, location_path = temp_paths
    db, catalog = setup_db(storage_path, catalog_path, location_path)
    planner = Planner()
    executor = Executor(db)

    collection = "consistency_collection"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=32, vector_type="float32")

    rng = np.random.default_rng(0)
    records = [{"text": f"record_{i}", "tags": [str(i % 2)]} for i in range(2)]
    vectors = [rng.random(32).astype(np.float32) for _ in records]

    # Prepare tasks as flat triples (idx, data, vec)
    tasks = [(i, records[i], vectors[i]) for i in range(len(records))]

    def task(args):
        idx, data, vec = args
        rec_id = executor.execute(
            planner.plan_insert(collection=collection,
                                original_data=data,
                                vector=vec,
                                metadata={"tags": data["tags"]})
        )
        # Filtered search by tag
        results = executor.execute(
            planner.plan_search(collection=collection,
                                query_vector=rng.random(32).astype(np.float32),
                                k=5,
                                where={"tags": data["tags"]})
        )
        assert any(res["id"] == rec_id for res in results)
        # Fetch
        rec = executor.execute(planner.plan_get(collection, rec_id))
        assert rec["original_data"] == data
        # Zero-distance check
        top = executor.execute(
            planner.plan_search(collection=collection,
                                query_vector=vec,
                                k=1)
        )[0]
        assert top["id"] == rec_id and np.isclose(top["distance"], 0.0)
        return rec_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(task, tasks))
    assert len(set(ids)) == len(records)

def test_insert_search_delete_and_fetch_consistency(temp_paths):
    storage_path, catalog_path, location_path = temp_paths
    db, catalog = setup_db(storage_path, catalog_path, location_path)
    planner = Planner()
    executor = Executor(db)

    collection = "isd_concurrent"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=24, vector_type="float32")

    rng = np.random.default_rng(1)
    tasks = [(i, rng.random(24).astype(np.float32), {"text": f"temp_{i}", "tags": ["temp"]}) for i in range(10)]

    def task(args):
        _, vec, data = args
        rec_id = executor.execute(
            planner.plan_insert(collection=collection,
                                original_data=data,
                                vector=vec,
                                metadata={"tags": data["tags"]})
        )
        # Pre-delete search
        pre = executor.execute(
            planner.plan_search(collection=collection,
                                query_vector=vec,
                                k=1)
        )
        assert pre and pre[0]["id"] == rec_id
        # Delete
        executor.execute(planner.plan_delete(collection=collection, record_id=rec_id))
        # Post-delete search
        post = executor.execute(
            planner.plan_search(collection=collection,
                                query_vector=vec,
                                k=1)
        )
        assert all(r["id"] != rec_id for r in post)
        # Fetch must fail
        with pytest.raises(Exception):
            executor.execute(planner.plan_get(collection, rec_id))
        return rec_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(task, tasks))
    assert len(set(ids)) == len(tasks)
