from typing import Optional, Set, Dict, Any, List, Tuple


class IdMapper:
    """
    Manages the mapping between user-provided string record IDs and
    internal integer IDs required by index implementations.

    This component encapsulates the dual-ID system logic, providing a clean
    separation of concerns from the actual vector indexing implementation.
    """

    def __init__(self):
        """Initialize the ID mapper with empty mappings."""
        # Maps user-provided string record IDs to internal integer IDs
        self._str_id_to_internal: Dict[str, int] = {}

        # Maps internal integer IDs back to user-provided string record IDs
        self._internal_to_str_id: Dict[int, str] = {}

        # Set of internal IDs that have been deleted and can be reused
        self._deleted_internal_ids: Set[int] = set()

        # Next available internal ID (for sequential assignment)
        self._next_internal_id: int = 0

    def add_mapping(self, record_id: str) -> int:
        """
        Create a mapping for a new record ID.

        Args:
            record_id: User-provided string record ID

        Returns:
            The assigned internal integer ID
        """
        # Check if record_id already exists
        if record_id in self._str_id_to_internal:
            return self._str_id_to_internal[record_id]

        # Assign internal ID (reuse a deleted ID if available)
        if self._deleted_internal_ids:
            internal_id = self._deleted_internal_ids.pop()
        else:
            internal_id = self._next_internal_id
            self._next_internal_id += 1

        # Store the mappings
        self._str_id_to_internal[record_id] = internal_id
        self._internal_to_str_id[internal_id] = record_id

        return internal_id

    def update_mapping(self, record_id: str) -> int:
        """
        Get the internal ID for a record ID, creating a new mapping if needed.
        If the record ID already exists, returns its current internal ID.

        Args:
            record_id: User-provided string record ID

        Returns:
            The internal integer ID
        """
        return self.add_mapping(record_id)

    def get_internal_id(self, record_id: str) -> Optional[int]:
        """
        Get the internal ID for a record ID.

        Args:
            record_id: User-provided string record ID

        Returns:
            The internal integer ID or None if not found
        """
        return self._str_id_to_internal.get(record_id)

    def get_record_id(self, internal_id: int) -> Optional[str]:
        """
        Get the record ID for an internal ID.

        Args:
            internal_id: Internal integer ID

        Returns:
            The user-provided string record ID or None if not found
        """
        return self._internal_to_str_id.get(internal_id)

    def delete_mapping(self, record_id: str) -> Optional[int]:
        """
        Delete a mapping and mark the internal ID as reusable.

        Args:
            record_id: User-provided string record ID

        Returns:
            The internal ID that was deleted, or None if not found
        """
        if record_id not in self._str_id_to_internal:
            return None

        internal_id = self._str_id_to_internal[record_id]

        # Remove both mappings
        del self._str_id_to_internal[record_id]
        del self._internal_to_str_id[internal_id]

        # Mark the internal ID as reusable
        self._deleted_internal_ids.add(internal_id)

        return internal_id

    def convert_to_internal_ids(self, record_ids: Set[str]) -> Set[int]:
        """
        Convert a set of record IDs to internal IDs.

        Args:
            record_ids: Set of user-provided string record IDs

        Returns:
            Set of corresponding internal integer IDs (only those that exist)
        """
        internal_ids = set()
        for record_id in record_ids:
            if record_id in self._str_id_to_internal:
                internal_ids.add(self._str_id_to_internal[record_id])
        return internal_ids

    def convert_to_record_ids(self, internal_ids: List[int]) -> List[str]:
        """
        Convert a list of internal IDs to record IDs.

        Args:
            internal_ids: List of internal integer IDs

        Returns:
            List of corresponding user-provided string record IDs (only those that exist)
        """
        record_ids = []
        for internal_id in internal_ids:
            if internal_id in self._internal_to_str_id:
                record_ids.append(self._internal_to_str_id[internal_id])
        return record_ids

    def has_record_id(self, record_id: str) -> bool:
        """
        Check if a record ID exists in the mapping.

        Args:
            record_id: User-provided string record ID

        Returns:
            True if the record ID exists, False otherwise
        """
        return record_id in self._str_id_to_internal

    def get_all_record_ids(self) -> List[str]:
        """
        Get all record IDs in the mapping.

        Returns:
            List of all user-provided string record IDs
        """
        return list(self._str_id_to_internal.keys())

    def count(self) -> int:
        """
        Get the number of mappings.

        Returns:
            The number of id mappings
        """
        return len(self._str_id_to_internal)

    def clear(self) -> None:
        """Clear all mappings and reset the next internal ID."""
        self._str_id_to_internal.clear()
        self._internal_to_str_id.clear()
        self._deleted_internal_ids.clear()
        self._next_internal_id = 0