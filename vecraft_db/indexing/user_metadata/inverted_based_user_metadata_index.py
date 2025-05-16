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
