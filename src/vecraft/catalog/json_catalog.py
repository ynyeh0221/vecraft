import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict

from src.vecraft.core.catalog_interface import Catalog
from src.vecraft.data.checksummed_data import CollectionSchema
from src.vecraft.data.exception import CollectionNotExistedException


class JsonCatalog(Catalog):
    """
    A JSON-based implementation of the Catalog interface that stores collection metadata in a JSON file.

    This class provides functionality to manage vector collection schemas with built-in fault tolerance:
    - Atomic file writes to prevent corruption during updates
    - Versioned backups for recovery
    - Consistency validation when loading
    - Recovery mechanisms from backup files

    Attributes:
        _path (Path): Path to the JSON catalog file.
        _collections (Dict[str, CollectionSchema]): Dictionary of collection schemas indexed by name.
        _max_backups (int): Maximum number of backup versions to keep.
    """

    def __init__(self, path: str = 'catalog.json', max_backups: int = 5):
        """
        Initialize a new JsonCatalog with fault tolerance features.

        Args:
            path (str, optional): Path to the JSON catalog file. Defaults to 'catalog.json'.
            max_backups (int, optional): Maximum number of backup versions to keep. Defaults to 5.
        """
        self._path = Path(path)
        self._collections: Dict[str, CollectionSchema] = {}
        self._max_backups = max_backups

        # Create directory if it doesn't exist
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._load()

    def _load(self):
        """
        Load collection schemas from the JSON file with validation and recovery.

        If the primary file is corrupted, attempts to recover from the most recent backup.
        """
        if self._path.exists():
            try:
                self._load_from_file(self._path)
                print(f"Successfully loaded catalog from {self._path}")
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"Error loading catalog from {self._path}: {e}")
                self._recover_from_backup()

    def _load_from_file(self, file_path: Path):
        """
        Load collection schemas from a specific file with validation.
        """
        content = file_path.read_text()
        data = json.loads(content)

        # Validate basic structure
        if not isinstance(data, dict):
            raise ValueError("Catalog file must contain a JSON object")

        # Clear current collections
        self._collections.clear()

        # Parse collections
        for col_name, schema_dict in data.items():
            if not isinstance(schema_dict, dict):
                raise ValueError(f"Invalid schema format for collection {col_name}")

            # Create a CollectionSchema - this will auto-compute the checksum
            schema = CollectionSchema(
                name=schema_dict.get('name', col_name),
                dim=schema_dict['dim'],
                vector_type=schema_dict['vector_type'],
                checksum_algorithm=schema_dict.get('checksum_algorithm', 'sha256')
            )

            # If there's a checksum in the file, validate it
            if 'checksum' in schema_dict:
                if schema.checksum != schema_dict['checksum']:
                    raise ValueError(f"Checksum mismatch for collection {col_name}")

            self._collections[col_name] = schema

    def _recover_from_backup(self):
        """
        Attempt to recover the catalog from the most recent valid backup.

        If no valid backups are found, initializes with an empty catalog.
        """
        backups = sorted(self._path.parent.glob(f'{self._path.stem}.bak.*'))

        # Try each backup from newest to oldest
        for backup in reversed(backups):
            try:
                print(f"Attempting to recover from backup: {backup}")
                self._load_from_file(backup)

                # If successful, restore this backup as the primary
                shutil.copy2(backup, self._path)
                print(f"Successfully recovered from backup: {backup}")
                return
            except Exception as e:
                print(f"Failed to recover from backup {backup}: {e}")

        # If we get here, no valid backups were found
        print("No valid backups found. Initializing with empty catalog.")
        self._collections = {}

    def _save(self):
        """
        Save collection schemas to the JSON file using atomic operations and versioning.

        This method:
        1. Creates a backup of the current file (if it exists)
        2. Writes data to a temporary file
        3. Atomically replaces the primary file with the temporary file
        4. Removes old backups exceeding the maximum number to keep
        """
        # Create backup of current file if it exists
        if self._path.exists():
            backup_path = self._path.with_suffix(f'.bak.{int(time.time())}')
            shutil.copy2(self._path, backup_path)

            # Clean up old backups
            self._cleanup_old_backups()

        # Prepare the data
        data = {name: schema.to_dict() for name, schema in self._collections.items()}

        # Create a temporary file
        temp_path = self._path.with_suffix('.tmp')
        temp_path.write_text(json.dumps(data, indent=2))
        # Ensure data is written to disk
        with open(str(temp_path), 'r') as f:
            os.fsync(f.fileno())

        # Atomically replace the original file
        if hasattr(os, 'replace'):  # Python 3.3+
            os.replace(temp_path, self._path)
        else:
            # Fallback for older Python versions
            try:
                os.remove(self._path)
            except FileNotFoundError:
                pass
            os.rename(temp_path, self._path)

    def _cleanup_old_backups(self):
        """
        Remove old backup files exceeding the maximum number to keep.
        """
        backups = sorted(self._path.parent.glob(f'{self._path.stem}.bak.*'))
        for old_backup in backups[:-self._max_backups]:
            try:
                old_backup.unlink()
                print(f"Removed old backup: {old_backup}")
            except Exception as e:
                print(f"Failed to remove old backup {old_backup}: {e}")

    def create_collection(self, collection_schema: CollectionSchema) -> None:
        """
        Create a new collection in the catalog.

        Args:
            name (str): Name of the collection.
            dim (int): Dimensionality of vectors in the collection.
            vector_type (str): Data type of the vector elements (e.g., 'float', 'int').
        """
        self._collections[collection_schema.name] = collection_schema
        self._save()

    def drop_collection(self, name: str) -> None:
        """
        Remove a collection from the catalog.

        Args:
            name (str): Name of the collection to drop.
        """
        self._collections.pop(name, None)
        self._save()

    def list_collections(self) -> List[CollectionSchema]:
        """
        List all collection names in the catalog.

        Returns:
            List[str]: Names of all collections in the catalog.
        """
        return list(self._collections.values())

    def get_schema(self, name: str) -> CollectionSchema:
        """
        Get the schema for a specific collection.

        Args:
            name (str): Name of the collection.

        Returns:
            CollectionSchema: Schema of the requested collection.

        Raises:
            CollectionNotExistedException: If the collection doesn't exist in the catalog.
        """
        if name not in self._collections:
            raise CollectionNotExistedException(f"Collection {name} not found", name)
        return self._collections[name]

    def verify_integrity(self) -> bool:
        """
        Verify the integrity of the catalog file.
        """
        if not self._path.exists():
            return False

        try:
            with open(self._path, 'r') as f:
                data = json.load(f)

            # Basic validation
            if not isinstance(data, dict):
                return False

            for col_name, schema_dict in data.items():
                # Validate each collection schema has required fields
                if (not isinstance(schema_dict, dict) or
                        'name' not in schema_dict or
                        'dim' not in schema_dict or
                        'vector_type' not in schema_dict or
                        'checksum_algorithm' not in schema_dict or
                        'checksum' not in schema_dict):
                    return False

                # Additional validation: try to reconstruct and verify checksum
                try:
                    schema = CollectionSchema(
                        name=schema_dict['name'],
                        dim=schema_dict['dim'],
                        vector_type=schema_dict['vector_type'],
                        checksum_algorithm=schema_dict['checksum_algorithm']
                    )
                    if schema.checksum != schema_dict['checksum']:
                        return False
                except Exception:
                    return False

            return True
        except Exception:
            return False