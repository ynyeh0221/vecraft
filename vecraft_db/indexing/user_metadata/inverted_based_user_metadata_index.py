"""
InvertedIndexMetadataIndex Workflow Diagram
===========================================

Dual-Index System: Supporting Both Equality and Range Queries on Metadata

DATA STRUCTURES OVERVIEW:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INDEX ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        EQUALITY INDEX                                   │    │
│  │                  _eq_index: Dict[str, Dict[Any, Set[str]]]              │    │
│  │                                                                         │    │
│  │  Structure: field_name -> value -> set_of_record_ids                    │    │
│  │                                                                         │    │
│  │  Example:                                                               │    │
│  │  {                                                                      │    │
│  │    "category": {                                                        │    │
│  │      "electronics": {"rec1", "rec5", "rec9"},                           │    │
│  │      "books": {"rec2", "rec7"},                                         │    │
│  │      "clothing": {"rec3", "rec8"}                                       │    │
│  │    },                                                                   │    │
│  │    "brand": {                                                           │    │
│  │      "apple": {"rec1", "rec4"},                                         │    │
│  │      "nike": {"rec3", "rec8"}                                           │    │
│  │    }                                                                    │    │
│  │  }                                                                      │    │
│  │                                                                         │    │
│  │  Optimized for: O(1) equality lookups, $in queries                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         RANGE INDEX                                     │    │
│  │                _range_index: Dict[str, List[Tuple[Any, str]]]           │    │
│  │                                                                         │    │
│  │  Structure: field_name -> sorted_list_of_(value, record_id)_tuples      │    │
│  │                                                                         │    │
│  │  Example:                                                               │    │
│  │  {                                                                      │    │
│  │    "price": [                                                           │    │
│  │      (10.99, "rec2"), (15.50, "rec7"), (25.00, "rec1"),                 │    │
│  │      (89.99, "rec5"), (120.00, "rec9")                                  │    │
│  │    ],                                                                   │    │
│  │    "rating": [                                                          │    │
│  │      (3.2, "rec8"), (4.1, "rec3"), (4.5, "rec1"),                       │    │
│  │      (4.8, "rec7"), (5.0, "rec2")                                       │    │
│  │    ]                                                                    │    │
│  │  }                                                                      │    │
│  │                                                                         │    │
│  │  Optimized for: O(log n) range queries, sorted iteration                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

INDEXING WORKFLOW (ADD OPERATION):
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RECORD INDEXING PROCESS                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────────────────────┐    │
│  │ MetadataPacket  │───▶│                VALIDATION                        │    │
│  │ record_id: "r1" │    │                                                  │    │
│  │ metadata: {     │    │  • Validate checksum                             │    │
│  │   "category":   │    │  • Extract record_id and metadata                │    │
│  │    "electronics"│    └──────────────────────────────────────────────────┘    │
│  │   "price": 25.0 │                           │                                │
│  │   "tags": ["new"│                           ▼                                │
│  │           "hot"]│    ┌──────────────────────────────────────────────────┐    │
│  │ }               │    │           FIELD PROCESSING                       │    │
│  └─────────────────┘    │                                                  │    │
│                         │  For each field, value pair in metadata:         │    │
│                         │                                                  │    │
│                         │  1. Normalize values to iterable:                │    │
│                         │     • Single value → (value,)                    │    │
│                         │     • List/set/tuple → iterate all values        │    │
│                         │                                                  │    │
│                         │  2. Process each normalized value:               │    │
│                         │     ├─ Add to equality index                     │    │
│                         │     └─ Try to add to range index (if sortable)   │    │
│                         └──────────────────────────────────────────────────┘    │
│                                                │                                │
│                                                ▼                                │
│                         ┌──────────────────────────────────────────────────┐    │
│                         │         DUAL INDEX UPDATES                       │    │
│                         │                                                  │    │
│                         │  EQUALITY INDEX UPDATE:                          │    │
│                         │  ┌─────────────────────────────────────────────┐ │    │
│                         │  │ _eq_index[field][value].add(record_id)      │ │    │
│                         │  │                                             │ │    │
│                         │  │ Example:                                    │ │    │
│                         │  │ _eq_index["category"]["electronics"]        │ │    │
│                         │  │   .add("r1")                                │ │    │
│                         │  │ _eq_index["tags"]["new"].add("r1")          │ │    │
│                         │  │ _eq_index["tags"]["hot"].add("r1")          │ │    │
│                         │  └─────────────────────────────────────────────┘ │    │
│                         │                                                  │    │
│                         │  RANGE INDEX UPDATE:                             │    │
│                         │  ┌─────────────────────────────────────────────┐ │    │
│                         │  │ try:                                        │ │    │
│                         │  │   insort(_range_index[field],               │ │    │
│                         │  │          (value, record_id))                │ │    │
│                         │  │ except TypeError:                           │ │    │
│                         │  │   # Skip non-sortable types                 │ │    │
│                         │  │                                             │ │    │
│                         │  │ Example:                                    │ │    │
│                         │  │ insort(_range_index["price"],               │ │    │
│                         │  │        (25.0, "r1"))                        │ │    │
│                         │  └─────────────────────────────────────────────┘ │    │
│                         └──────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

QUERYING WORKFLOW:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              QUERY PROCESSING PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────────────────────┐    │
│  │ Query Filter:   │───▶│              QUERY PARSING                       │    │
│  │ {               │    │                                                  │    │
│  │  "category":    │    │  Input: where = Dict[str, Any]                   │    │
│  │    "electronics"│    │                                                  │    │
│  │  "price": {     │    │  For each field, condition pair:                 │    │
│  │    "$gte": 20,  │    │  └─ Determine query type and dispatch            │    │
│  │    "$lt": 100   │    └──────────────────────────────────────────────────┘    │
│  │  }              │                           │                                │
│  │ }               │                           ▼                                │
│  └─────────────────┘    ┌──────────────────────────────────────────────────┐    │
│                         │         CONDITION PROCESSING                     │    │
│                         │                                                  │    │
│                         │  ┌─ EQUALITY QUERY ─────────────────────────────┐│    │
│                         │  │  Pattern: field: value                       ││    │
│                         │  │  Example: "category": "electronics"          ││    │
│                         │  │                                              ││    │
│                         │  │  Process:                                    ││    │
│                         │  │  └─ return _eq_index[field].get(value, set())││    │
│                         │  └──────────────────────────────────────────────┘│    │
│                         │                                                  │    │
│                         │  ┌─ $IN QUERY ──────────────────────────────────┐│    │
│                         │  │  Pattern: field: {"$in": [v1, v2, v3]}       ││    │
│                         │  │  Example: "category": {"$in": ["books",      ││    │
│                         │  │                               "electronics"]}││    │
│                         │  │                                              ││    │
│                         │  │  Process:                                    ││    │
│                         │  │  └─ Union all _eq_index[field][vi] sets      ││    │
│                         │  └──────────────────────────────────────────────┘│    │
│                         │                                                  │    │
│                         │  ┌─ RANGE QUERY ────────────────────────────────┐│    │
│                         │  │  Pattern: field: {"$gte": X, "$lt": Y, ...}  ││    │
│                         │  │  Operators: $gte, $gt, $lte, $lt             ││    │
│                         │  │                                              ││    │
│                         │  │  Process: (See detailed flow below)          ││    │
│                         │  └──────────────────────────────────────────────┘│    │
│                         └──────────────────────────────────────────────────┘    │
│                                                 │                               │
│                                                 ▼                               │
│                         ┌──────────────────────────────────────────────────┐    │
│                         │           RESULT INTERSECTION                    │    │
│                         │                                                  │    │
│                         │  Logic: AND all conditions together              │    │
│                         │                                                  │    │
│                         │  result_ids = None                               │    │
│                         │  for each condition:                             │    │
│                         │    condition_ids = process_condition()           │    │
│                         │    if condition_ids is empty:                    │    │
│                         │      return empty_set()  # Short-circuit         │    │
│                         │    if result_ids is None:                        │    │
│                         │      result_ids = condition_ids                  │    │
│                         │    else:                                         │    │
│                         │      result_ids = result_ids ∩ condition_ids     │    │
│                         │      if result_ids is empty:                     │    │
│                         │        return empty_set()  # Short-circuit       │    │
│                         │                                                  │    │
│                         │  return result_ids                               │    │
│                         └──────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

RANGE QUERY DETAILED PROCESSING:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            RANGE QUERY ALGORITHM                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Input: field: {"$gte": 20, "$lt": 100}                                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 1: BOUND COMPUTATION                            │    │
│  │                                                                         │    │
│  │  Determine search bounds for binary search:                             │    │
│  │  • low = cond.get("$gte", cond.get("$gt"))                              │    │
│  │  • high = cond.get("$lte", cond.get("$lt"))                             │    │
│  │                                                                         │    │
│  │  Example: low = 20, high = 100                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                        │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 2: BINARY SEARCH SLICE                          │    │
│  │                                                                         │    │
│  │  Given sorted list: [(10, "r2"), (25, "r1"), (89, "r5"), (120, "r9")]   │    │
│  │                                                                         │    │
│  │  start = bisect_left(lst, (20, ""))     # Find insertion point ≥ 20     │    │
│  │  end = bisect_right(lst, (100, chr(255))) # Find insertion point > 100  │    │
│  │                                                                         │    │
│  │  Result slice: [(25, "r1"), (89, "r5")]                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                        │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                  STEP 3: STRICT BOUND FILTERING                         │    │
│  │                                                                         │    │
│  │  For each (value, record_id) in slice:                                  │    │
│  │    Check strict inequalities that bisect can't handle:                  │    │
│  │                                                                         │    │
│  │    if "$gt" in cond and value <= cond["$gt"]:                           │    │
│  │      continue  # Skip this record                                       │    │
│  │    if "$lt" in cond and value >= cond["$lt"]:                           │    │
│  │      continue  # Skip this record                                       │    │
│  │                                                                         │    │
│  │    results.add(record_id)  # Record passes all conditions               │    │
│  │                                                                         │    │
│  │  Example with "$gte": 20, "$lt": 100:                                   │    │
│  │  • (25, "r1"): 25 >= 20 ✓, 25 < 100 ✓ → Include "r1"                    │    │
│  │  • (89, "r5"): 89 >= 20 ✓, 89 < 100 ✓ → Include "r5"                    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

UPDATE AND DELETE OPERATIONS:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODIFICATION OPERATIONS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  UPDATE OPERATION:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  def update(old_item, new_item):                                        │    │
│  │    1. Validate both items                                               │    │
│  │    2. delete(old_item)    # Remove old metadata                         │    │
│  │    3. add(new_item)       # Add new metadata                            │    │
│  │                                                                         │    │
│  │  Two-phase approach ensures atomic update                               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  DELETE OPERATION:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  For each field, value in metadata:                                     │    │
│  │                                                                         │    │
│  │  EQUALITY INDEX CLEANUP:                                                │    │
│  │  └─ _eq_index[field][value].discard(record_id)                          │    │
│  │                                                                         │    │
│  │  RANGE INDEX CLEANUP:                                                   │    │
│  │  1. Find position range: bisect_left/right((value, record_id))          │    │
│  │  2. Search within range for exact (value, record_id) match              │    │
│  │  3. Remove the matching tuple from sorted list                          │    │
│  │                                                                         │    │
│  │  Note: Range deletion is O(n) in worst case due to list.pop()           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

KEY DESIGN PRINCIPLES:
• Dual indexing: Optimized for both equality (O(1)) and range (O(log n)) queries
• Multi-value support: Handles list/set/tuple metadata values naturally
• Early termination: Short-circuits when any condition yields empty results
• Type safety: Gracefully handles non-sortable types in range index
• Memory efficiency: Shared record IDs across multiple index entries
• Query flexibility: Supports complex combinations of equality, IN, and range conditions
"""
import pickle
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from typing import Any, Dict, Set, Optional, Tuple, List

from vecraft_data_model.index_packets import MetadataPacket
from vecraft_db.core.interface.user_metadata_index_interface import MetadataIndexInterface


class InvertedIndexMetadataIndex(MetadataIndexInterface):
    """
    A metadata record_vector supporting equality and range queries.

    - Equality queries via inverted record_vector: field -> value -> set(record_id).
    - Range queries via sorted lists: field -> list of (value, record_id).
    """
    def __init__(self):
        self._eq_index: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._range_index: Dict[str, list] = defaultdict(list)

    def add(self, item: MetadataPacket) -> None:
        """
        Index a record's metadata.
        """
        # Validate upfront
        item.validate_checksum()
        rid = item.record_id

        # Nothing to do if no metadata
        metadata = item.metadata or {}
        for field, value in metadata.items():
            # Normalize to an iterable of values
            if isinstance(value, (list, set, tuple)):
                values = value
            else:
                values = (value,)

            eq_bucket = self._eq_index[field]
            range_list = self._range_index[field]

            for v in values:
                eq_bucket[v].add(rid)
                # Only index in the range list if it's sortable
                try:
                    insort(range_list, (v, rid))
                except TypeError:
                    # skip non-order able types
                    continue

    def update(self, old_item: MetadataPacket, new_item: MetadataPacket) -> None:
        """
        Update a record's metadata by removing old and adding new.
        """
        old_item.validate_checksum()
        new_item.validate_checksum()
        self.delete(old_item)
        self.add(new_item)
        old_item.validate_checksum()
        new_item.validate_checksum()

    def delete(self, item: MetadataPacket) -> None:
        """
        Remove a record's metadata from the record_vector.
        """
        item.validate_checksum()
        rid = item.record_id
        metadata = item.metadata or {}
        for field, value in metadata.items():
            if isinstance(value, (list, set, tuple)):
                for v in value:
                    self._eq_index[field][v].discard(rid)
                    self._remove_from_range(field, v, rid)
            else:
                self._eq_index[field][value].discard(rid)
                self._remove_from_range(field, value, rid)
        item.validate_checksum()

    def _remove_from_range(self, field: str, value: Any, rid: str) -> None:
        lst = self._range_index[field]
        lo = bisect_left(lst, (value, rid))
        hi = bisect_right(lst, (value, rid))
        for i in range(lo, hi):
            if lst[i][1] == rid:
                lst.pop(i)
                break

    def get_matching_ids(self, where: Dict[str, Any]) -> Optional[Set[str]]:
        """
        Return IDs matching filter conditions. Supports:
        - equality:     field: value
        - $in:          field: {"$in": [v1, v2]}
        - range:        field: {"$gte": low, "$lte": high, "$gt": low2, "$lt": high2}

        Returns empty set if no matches; None only if `where` is empty.
        """
        result_ids: Optional[Set[str]] = None

        for field, cond in where.items():
            ids = self._ids_for_condition(field, cond)
            if not ids:
                return set()
            result_ids = ids if result_ids is None else result_ids & ids
            if not result_ids:
                return set()

        return result_ids

    def _ids_for_condition(self, field: str, cond: Any) -> Set[str]:
        """Dispatch to the right matcher based on whether `cond` is a dict."""
        if not isinstance(cond, dict):
            return self._eq_ids(field, cond)
        ids = set()
        ids |= self._in_ids(field, cond)
        ids |= self._range_ids(field, cond)
        return ids

    def _eq_ids(self, field: str, value: Any) -> Set[str]:
        """Simple equality lookup."""
        return set(self._eq_index[field].get(value, []))

    def _in_ids(self, field: str, cond: Dict[str, Any]) -> Set[str]:
        """Handle a {"$in": [...]} clause (or return empty set)."""
        values = cond.get("$in")
        if not values:
            return set()
        ids: Set[str] = set()
        for v in values:
            ids |= set(self._eq_index[field].get(v, []))
        return ids

    def _range_ids(self, field: str, cond: Dict[str, Any]) -> Set[str]:
        """
        Handle any of $gte, $gt, $lte, $lt in a single pass over the sorted list.
        """
        if not any(op in cond for op in ("$gte", "$gt", "$lte", "$lt")):
            return set()

        lst: List[Tuple[Any, str]] = self._range_index[field]
        low, high = self._compute_bounds(cond)
        start = bisect_left(lst, (low, "")) if low is not None else 0
        end = bisect_right(lst, (high, chr(255))) if high is not None else len(lst)

        results: Set[str] = set()
        for val, rid in lst[start:end]:
            if self._violates_strict_bounds(val, cond):
                continue
            results.add(rid)
        return results

    def _compute_bounds(self, cond: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Determine the non-strict bounds for slicing.
        Uses $gte as low when present, else $gt;
        uses $lte as high when present, else $lt.
        """
        low = cond.get("$gte", cond.get("$gt"))
        high = cond.get("$lte", cond.get("$lt"))
        return low, high

    def _violates_strict_bounds(self, val: Any, cond: Dict[str, Any]) -> bool:
        """
        Check strict inequalities ($gt, $lt) that can't be handled by the slice.
        """
        if "$gt" in cond and val <= cond["$gt"]:
            return True
        if "$lt" in cond and val >= cond["$lt"]:
            return True
        return False

    def serialize(self) -> bytes:
        """
        Serialize the metadata record_vector to bytes for snapshotting.
        """
        state = {
            'eq_index': {field: dict(vals) for field, vals in self._eq_index.items()},
            'range_index': {field: list(lst) for field, lst in self._range_index.items()}
        }
        return pickle.dumps(state)

    def deserialize(self, data: bytes) -> None:
        """
        Restore the metadata record_vector from serialized bytes.
        """
        state = pickle.loads(data)
        self._eq_index = defaultdict(lambda: defaultdict(set), {
            field: defaultdict(set, {val: set(rids) for val, rids in vals.items()})
            for field, vals in state['eq_index'].items()
        })
        self._range_index = defaultdict(list, {
            field: list(lst)
            for field, lst in state['range_index'].items()
        })
