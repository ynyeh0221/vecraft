import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from src.vecraft.core.data import DataPacket, QueryPacket
from src.vecraft.core.errors import RecordNotFoundError
from src.vecraft.engine.vector_db import VectorDB
from src.vecraft.index.record_metadata.metadata_index import MetadataIndex
from src.vecraft.index.record_vector.hnsw import HNSW
from src.vecraft.metadata.catalog import JsonCatalog
from src.vecraft.query.executor import Executor
from src.vecraft.query.planner import Planner
from src.vecraft.storage.mmap_json_storage_index_engine import MMapJsonStorageIndexEngine
from src.vecraft.wal.wal_manager import WALManager


def setup_db(test_dir):
    """Setup database with all files stored in test_dir"""
    # Convert test_dir to Path object if it's not already
    test_dir = Path(test_dir)

    # Use Path.joinpath or / operator for path construction
    catalog_path = test_dir / "catalog.json"
    catalog = JsonCatalog(str(catalog_path))

    def wal_factory(wal_path: str):
        return WALManager(test_dir / wal_path)

    def storage_factory(data_path: str, index_path: str):
        return MMapJsonStorageIndexEngine(
            str(test_dir / data_path),
            str(test_dir / index_path)
        )

    def vector_index_factory(kind: str, dim: int):
        if kind == "hnsw":
            return HNSW(dim=dim, M=16, ef_construction=200)
        raise ValueError(f"Unknown index kind: {kind}")

    def metadata_index_factory():
        return MetadataIndex()

    db = VectorDB(catalog=catalog,
                  wal_factory=wal_factory,
                  storage_factory=storage_factory,
                  vector_index_factory=vector_index_factory,
                  metadata_index_factory=metadata_index_factory)
    return db, catalog


@pytest.fixture
def temp_paths(tmp_path):
    # Create a temporary directory for testing
    test_dir = tmp_path / "test_db"
    test_dir.mkdir()

    # Return the Path object, not just the name
    yield test_dir

    # Clean up the temporary directory
    shutil.rmtree(test_dir)


def test_insert_search_and_fetch_consistency(temp_paths):
    test_dir = temp_paths
    db, catalog = setup_db(test_dir)
    planner = Planner()
    executor = Executor(db)

    collection = "consistency_collection"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=32, vector_type="float32")

    rng = np.random.default_rng(0)
    records = [{"text": f"record_{i}", "tags": [str(i % 2)]} for i in range(20)]
    vectors = [rng.random(32).astype(np.float32) for _ in records]

    # Prepare tasks as flat triples (idx, data, vec)
    tasks = [(i, records[i], vectors[i]) for i in range(len(records))]

    def task(args):
        idx, data, vec = args
        rec_id = executor.execute(
            planner.plan_insert(collection=collection,
                                data_packet=DataPacket(type="insert",
                                                       original_data=data,
                                                       vector=vec,
                                                       metadata={"tags": data["tags"]},
                                                       record_id=str(idx)))
        )
        # Filtered search by tag
        results = executor.execute(
            planner.plan_search(collection=collection,
                                query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                         k=20,
                                                         where={"tags": data["tags"]}))
        )
        assert any(res["id"] == rec_id for res in results)
        # Fetch
        rec = executor.execute(planner.plan_get(collection, rec_id))
        assert rec["original_data"] == data
        # Zero-distance check
        top = executor.execute(
            planner.plan_search(collection=collection,
                                query_packet=QueryPacket(query_vector=vec, k=1))
        )[0]
        assert top["id"] == rec_id and np.isclose(top["distance"], 0.0)
        return rec_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(task, tasks))
    assert len(set(ids)) == len(records)

def test_insert_search_delete_and_fetch_consistency(temp_paths):
    test_dir = temp_paths
    db, catalog = setup_db(test_dir)
    planner = Planner()
    executor = Executor(db)

    collection = "isd_concurrent"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=24, vector_type="float32")

    rng = np.random.default_rng(1)
    tasks = [(i, rng.random(24).astype(np.float32), {"text": f"temp_{i}", "tags": ["temp"]}) for i in range(10)]

    def task(args):
        idx, vec, data = args
        rec_id = executor.execute(
            planner.plan_insert(collection=collection,
                                data_packet=DataPacket(type="insert",
                                                       original_data=data,
                                                       vector=vec,
                                                       metadata={"tags": data["tags"]},
                                                       record_id=str(idx)))
        )
        # Pre-delete search
        pre = executor.execute(
            planner.plan_search(collection=collection,
                                query_packet=QueryPacket(query_vector=vec, k=1))
        )
        assert pre and pre[0]["id"] == rec_id
        # Delete
        executor.execute(planner.plan_delete(collection=collection,
                                             data_packet=DataPacket(type="delete",
                                                                    record_id=str(idx))))
        # Post-delete search
        post = executor.execute(
            planner.plan_search(collection=collection,
                                query_packet=QueryPacket(query_vector=vec, k=1))
        )
        assert all(r["id"] != rec_id for r in post)
        # Fetch must fail
        with pytest.raises(Exception):
            executor.execute(planner.plan_get(collection, rec_id))
        return rec_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(task, tasks))
    assert len(set(ids)) == len(tasks)

def test_update_consistency(temp_paths):
    """Test updating records and ensuring consistency before and after updates."""
    test_dir = temp_paths
    db, catalog = setup_db(test_dir)
    planner = Planner()
    executor = Executor(db)

    collection = "update_consistency"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=32, vector_type="float32")

    rng = np.random.default_rng(0)

    # Create initial record
    original_data = {"text": "original", "tags": ["old"]}
    original_vector = rng.random(32).astype(np.float32)

    # Insert the record
    record_id = executor.execute(
        planner.plan_insert(collection=collection,
                            data_packet=DataPacket(type="insert",
                                                   original_data=original_data,
                                                   vector=original_vector,
                                                   metadata={"tags": original_data["tags"]},
                                                   record_id="id1"))
    )

    # Verify it's searchable and fetchable
    results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=original_vector,
                                                     k=5,
                                                     where={"tags": original_data["tags"]}))
    )
    assert any(res["id"] == record_id for res in results)
    rec = executor.execute(planner.plan_get(collection, record_id))
    assert rec["original_data"] == original_data

    # Update the record with new data and vector
    updated_data = {"text": "updated", "tags": ["new"]}
    updated_vector = rng.random(32).astype(np.float32)

    # Re-insert with same ID (update)
    executor.execute(
        planner.plan_insert(collection=collection,
                            data_packet=DataPacket(type="insert",
                                                   original_data=updated_data,
                                                   vector=updated_vector,
                                                   metadata={"tags": updated_data["tags"]},
                                                   record_id=record_id))
    )

    # Verify old metadata doesn't return the record
    old_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=5,
                                                     where={"tags": original_data["tags"]}))
    )
    assert all(res["id"] != record_id for res in old_results)

    # Verify new metadata returns the record
    new_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=5,
                                                     where={"tags": updated_data["tags"]}))
    )
    assert any(res["id"] == record_id for res in new_results)

    # Verify the updated record can be fetched
    updated_rec = executor.execute(planner.plan_get(collection, record_id))
    assert updated_rec["original_data"] == updated_data

    # Verify zero-distance search with updated vector works
    top = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=updated_vector, k=1))
    )[0]
    assert top["id"] == record_id and np.isclose(top["distance"], 0.0)


def test_batch_operations_consistency(temp_paths):
    """Test bulk operations (insert, search, delete) for consistency."""
    test_dir = temp_paths
    db, catalog = setup_db(test_dir)
    planner = Planner()
    executor = Executor(db)

    collection = "batch_consistency"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=32, vector_type="float32")

    rng = np.random.default_rng(0)

    # Prepare batch of records with two different tags
    batch_size = 50
    records = []
    vectors = []
    record_ids = []

    for i in range(batch_size):
        tag = "even" if i % 2 == 0 else "odd"
        records.append({"text": f"batch_{i}", "tags": [tag]})
        vectors.append(rng.random(32).astype(np.float32))

    # Batch insert (sequentially)
    for i in range(batch_size):
        rec_id = executor.execute(
            planner.plan_insert(collection=collection,
                                data_packet=DataPacket(type="insert",
                                                       original_data=records[i],
                                                       vector=vectors[i],
                                                       metadata={"tags": records[i]["tags"]},
                                                       record_id=str(i)))
        )
        record_ids.append(rec_id)

    # Verify all records can be fetched
    for i, rec_id in enumerate(record_ids):
        rec = executor.execute(planner.plan_get(collection, rec_id))
        assert rec["original_data"] == records[i]

    # Verify filtering by tag works
    even_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=batch_size,
                                                     where={"tags": ["even"]}))
    )
    assert len(even_results) > 0

    odd_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=batch_size,
                                                     where={"tags": ["odd"]}))
    )
    assert len(odd_results) > 0

    # Batch delete even records
    for i, rec_id in enumerate(record_ids):
        if i % 2 == 0:  # Delete even records
            executor.execute(planner.plan_delete(collection=collection,
                                                 data_packet=DataPacket(type="delete",
                                                                        record_id=str(rec_id))))

    # Verify even records are gone
    for i, rec_id in enumerate(record_ids):
        if i % 2 == 0:  # Even records should be gone
            with pytest.raises(RecordNotFoundError):
                executor.execute(planner.plan_get(collection, rec_id))
        else:  # Odd records should still be there
            rec = executor.execute(planner.plan_get(collection, rec_id))
            assert rec["original_data"] == records[i]

    # Verify searching by "even" tag returns no results
    empty_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=batch_size,
                                                     where={"tags": ["even"]}))
    )
    assert len(empty_results) == 0

    # Odd records should still be searchable
    odd_results_after = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=batch_size,
                                                     where={"tags": ["odd"]}))
    )
    assert len(odd_results_after) > 0

def test_complex_filtering_consistency(temp_paths):
    """Test complex filtering scenarios (multiple tags, nested conditions)."""
    test_dir = temp_paths
    db, catalog = setup_db(test_dir)
    planner = Planner()
    executor = Executor(db)

    collection = "filtering_collection"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=32, vector_type="float32")

    rng = np.random.default_rng(0)

    # Create records with different combinations of tags
    records = [
        {"text": "red apple", "tags": ["red", "fruit", "sweet"], "category": "produce", "count": 5},
        {"text": "green apple", "tags": ["green", "fruit", "sour"], "category": "produce", "count": 3},
        {"text": "red strawberry", "tags": ["red", "fruit", "sweet"], "category": "produce", "count": 10},
        {"text": "red firetruck", "tags": ["red", "vehicle", "emergency"], "category": "toys", "count": 1},
        {"text": "blue car", "tags": ["blue", "vehicle"], "category": "toys", "count": 2}
    ]

    record_ids = []
    for idx, data in enumerate(records):
        vec = rng.random(32).astype(np.float32)
        rec_id = executor.execute(
            planner.plan_insert(collection=collection,
                                data_packet=DataPacket(type="insert",
                                                       original_data=data,
                                                       vector=vec,
                                                       metadata={
                                                           "tags": data["tags"],
                                                           "category": data["category"],
                                                           "count": data["count"]
                                                       },
                                                       record_id=str(idx)
                                                       ))
        )
        record_ids.append(rec_id)

    # Test filtering by single tag
    red_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=10,
                                                     where={"tags": ["red"]}))
    )
    assert len(red_results) == 3  # 3 red items

    # Test filtering by category
    produce_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=10,
                                                     where={"category": "produce"}))
    )
    assert len(produce_results) == 3  # 3 produce items

    # Test filtering with numeric comparison
    high_count_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=10,
                                                     where={"count": {"$gt": 4}}))
    )
    assert len(high_count_results) == 2  # 2 items with count > 4

    # Test combined filtering
    red_produce_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=rng.random(32).astype(np.float32),
                                                     k=10,
                                                     where={"tags": ["red"], "category": "produce"}))
    )
    assert len(red_produce_results) == 2  # 2 red produce items


def test_concurrent_modifications(temp_paths):
    """Test concurrent modifications to the same records."""
    test_dir = temp_paths
    db, catalog = setup_db(test_dir)
    planner = Planner()
    executor = Executor(db)

    collection = "concurrent_mod"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=32, vector_type="float32")

    rng = np.random.default_rng(0)

    # Insert an initial record
    initial_data = {"text": "initial", "version": 0}
    initial_vector = rng.random(32).astype(np.float32)
    record_id = executor.execute(
        planner.plan_insert(collection=collection,
                            data_packet=DataPacket(type="insert",
                                                   original_data=initial_data,
                                                   vector=initial_vector,
                                                   metadata={"version": initial_data["version"]},
                                                   record_id="id1"
                                                   ))
    )

    # Define task for concurrent updates
    def update_task(i):
        data = {"text": f"update_{i}", "version": i}
        vec = rng.random(32).astype(np.float32)
        try:
            executor.execute(
                planner.plan_insert(collection=collection,
                                    data_packet=DataPacket(type="insert",
                                                           original_data=data,
                                                           vector=vec,
                                                           metadata={"version": data["version"]},
                                                           record_id=record_id
                                                           ))
            )
            return i
        except Exception as e:
            return f"Error: {e}"

    # Run concurrent updates
    num_updates = 10
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(update_task, range(1, num_updates + 1)))

    # Verify record exists and has been updated
    final_record = executor.execute(planner.plan_get(collection, record_id))
    assert final_record is not None
    assert final_record["original_data"]["version"] > 0

    # The final version should be from one of our updates
    assert any(final_record["original_data"]["version"] == i for i in range(1, num_updates + 1))

def test_special_characters(temp_paths):
    """Test with special characters in data."""
    test_dir = temp_paths
    db, catalog = setup_db(test_dir)
    planner = Planner()
    executor = Executor(db)

    collection = "special_chars"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=8, vector_type="float32")

    special_data = {
        "text": "Special chars: !@#$%^&*()",
        "nested": {
            "array": [1, 2, 3],
            "emoji": "😀🚀🌍",
            "quotes": "\"quoted text\""
        }
    }

    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    special_vec = rng.random(8).astype(np.float32)
    record_id = executor.execute(
        planner.plan_insert(collection=collection,
                            data_packet=DataPacket(type="insert",
                                                   original_data=special_data,
                                                   vector=special_vec,
                                                   metadata={"has_emoji": True},
                                                   record_id="id1"
                                                   ))
    )

    # Verify complex data is preserved
    result = executor.execute(planner.plan_get(collection=collection, record_id=record_id))
    assert result["original_data"]["nested"]["emoji"] == "😀🚀🌍"
    assert result["original_data"]["nested"]["quotes"] == "\"quoted text\""

def test_large_metadata(temp_paths):
    """Test with a large metadata object."""
    test_dir = temp_paths
    db, catalog = setup_db(test_dir)
    planner = Planner()
    executor = Executor(db)

    collection = "large_meta"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=4, vector_type="float32")

    # Create large metadata with many keys
    large_meta = {}
    for i in range(100):
        large_meta[f"key_{i}"] = f"value_{i}"

    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    large_vec = rng.random(4).astype(np.float32)
    record_id = executor.execute(
        planner.plan_insert(collection=collection,
                            data_packet=DataPacket(type="insert",
                                                   original_data={"text": "large_meta"},
                                                   vector=large_vec,
                                                   metadata=large_meta,
                                                   record_id="id1"
                                                   ))
    )

    # Verify we can search by one of the metadata values
    results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=large_vec,
                                                     k=1,
                                                     where={"key_50": "value_50"}))
    )
    assert len(results) == 1
    assert results[0]["id"] == record_id

def test_scalability(temp_paths):
    """Test database performance with a larger number of records."""
    test_dir = temp_paths
    db, catalog = setup_db(test_dir)
    planner = Planner()
    executor = Executor(db)

    collection = "scalability"
    if collection not in catalog.list_collections():
        catalog.create_collection(collection, dim=32, vector_type="float32")

    rng = np.random.default_rng(0)

    # Insert a larger number of records
    num_records = 200  # Adjust based on your system's capabilities
    record_ids = []

    # Create batches of different categories
    categories = ["electronics", "books", "clothing", "food", "sports"]

    start_time = time.time()

    for i in range(num_records):
        category = categories[i % len(categories)]
        data = {
            "title": f"Item {i}",
            "category": category,
            "price": float(rng.integers(10, 1000))
        }

        vec = rng.random(32).astype(np.float32)

        rec_id = executor.execute(
            planner.plan_insert(collection=collection,
                                data_packet=DataPacket(type="insert",
                                                       original_data=data,
                                                       vector=vec,
                                                       metadata={
                                                           "category": category,
                                                           "price_range": "high" if data["price"] > 500 else "low"
                                                       },
                                                       record_id=str(i)
                                                       ))
        )
        record_ids.append(rec_id)

    insert_time = time.time() - start_time
    print(f"Inserted {num_records} records in {insert_time:.2f} seconds")

    # Test search performance
    start_time = time.time()

    # Random vector similarity search
    search_vector = rng.random(32).astype(np.float32)
    results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=search_vector, k=10))
    )

    # Filtered search
    filtered_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=search_vector,
                                                     k=10,
                                                     where={"category": "electronics"}))
    )

    # Complex filtered search
    complex_results = executor.execute(
        planner.plan_search(collection=collection,
                            query_packet=QueryPacket(query_vector=search_vector,
                                                     k=10,
                                                     where={"category": "electronics", "price_range": "high"}))
    )

    search_time = time.time() - start_time
    print(f"Performed 3 searches in {search_time:.2f} seconds")

    # Assertions to verify functionality
    assert len(results) <= 10
    assert all(r["metadata"]["category"] == "electronics" for r in filtered_results)
    assert all(r["metadata"]["category"] == "electronics" and
               r["metadata"]["price_range"] == "high" for r in complex_results)