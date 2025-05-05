import pickle
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from typing import Any, Dict, Set, Optional

from src.vecraft.core.user_metadata_index_interface import MetadataIndexInterface
from src.vecraft.data.checksummed_data import MetadataItem, validate_checksum


class MetadataIndex(MetadataIndexInterface):
    """
    A user_metadata_index record_vector supporting equality and range queries.

    - Equality queries via inverted record_vector: field -> value -> set(record_id).
    - Range queries via sorted lists: field -> list of (value, record_id).
    """
    def __init__(self):
        self._eq_index: Dict[str, Dict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._range_index: Dict[str, list] = defaultdict(list)

    @validate_checksum
    def add(self, item: MetadataItem) -> None:
        """
        Index a record's user_metadata_index.
        """
        rid = item.record_id
        for field, value in item.metadata.items():
            if isinstance(value, (list, set, tuple)):
                for v in value:
                    self._eq_index[field][v].add(rid)
                    try:
                        insort(self._range_index[field], (v, rid))
                    except TypeError:
                        pass
            else:
                self._eq_index[field][value].add(rid)
                try:
                    insort(self._range_index[field], (value, rid))
                except TypeError:
                    pass

    @validate_checksum
    def update(self, old_item: MetadataItem, new_item: MetadataItem) -> None:
        """
        Update a record's user_metadata_index by removing old and adding new.
        """
        self.delete(old_item)
        self.add(new_item)

    @validate_checksum
    def delete(self, item: MetadataItem) -> None:
        """
        Remove a record's user_metadata_index from the record_vector.
        """
        rid = item.record_id
        for field, value in item.metadata.items():
            if isinstance(value, (list, set, tuple)):
                for v in value:
                    self._eq_index[field][v].discard(rid)
                    self._remove_from_range(field, v, rid)
            else:
                self._eq_index[field][value].discard(rid)
                self._remove_from_range(field, value, rid)

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
        - equality: field: value
        - $in:    field: {"$in": [v1, v2]}
        - range:  field: {"$gte": low, "$lte": high, "$gt": low2, "$lt": high2}

        Returns None if unable to use record_vector effectively.
        """
        result_ids = None

        for field, cond in where.items():
            ids = set()

            # simple equality
            if not isinstance(cond, dict):
                ids = set(self._eq_index[field].get(cond, []))

            else:
                # handle $in
                if "$in" in cond:
                    for v in cond["$in"]:
                        ids |= self._eq_index[field].get(v, set())

                # only perform a range scan if there's a range operator present
                if any(k in cond for k in ("$gte", "$gt", "$lte", "$lt")):
                    low = cond.get("$gte") if "$gte" in cond else cond.get("$gt")
                    high = cond.get("$lte") if "$lte" in cond else cond.get("$lt")

                    lst = self._range_index[field]
                    start = bisect_left(lst, (low, "")) if low is not None else 0
                    end = bisect_right(lst, (high, chr(255))) if high is not None else len(lst)

                    for val, rid in lst[start:end]:
                        if "$gt" in cond and val <= cond["$gt"]:
                            continue
                        if "$lt" in cond and val >= cond["$lt"]:
                            continue
                        ids.add(rid)

            # if no matches at all, shortcut to empty set
            if not ids:
                return set()

            # intersect with previous fields’ results
            result_ids = ids if result_ids is None else (result_ids & ids)
            if not result_ids:
                return set()

        return result_ids

    def serialize(self) -> bytes:
        """
        Serialize the user_metadata_index record_vector to bytes for snapshotting.
        """
        state = {
            'eq_index': {field: dict(vals) for field, vals in self._eq_index.items()},
            'range_index': {field: list(lst) for field, lst in self._range_index.items()}
        }
        return pickle.dumps(state)

    def deserialize(self, data: bytes) -> None:
        """
        Restore the user_metadata_index record_vector from serialized bytes.
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
