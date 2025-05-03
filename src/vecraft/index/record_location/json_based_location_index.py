import json
from pathlib import Path
from typing import Optional, Dict, List

from src.vecraft.index.record_location.location_index_interface import RecordLocationIndex


class JsonRecordLocationIndex(RecordLocationIndex):
    """
    JSON file–backed implementation of ConfigStore.
    """

    def __init__(self, path: Path):
        self._path = path
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            content = self._path.read_text()
            if content.strip():  # Check if file has non-whitespace content
                self._config = json.loads(content)
            else:
                # File exists but is empty, initialize default config
                self._config = {
                    'next_id': 0,
                    'records': {},
                    'deleted_records': []
                }
                self._save()
        else:
            # File doesn't exist, initialize default config
            self._config = {
                'next_id': 0,
                'records': {},
                'deleted_records': []
            }
            self._save()

    def _save(self) -> None:
        # atomic write could be added here if desired
        self._path.write_text(json.dumps(self._config, indent=2))

    def get_next_id(self) -> str:
        nid = self._config['next_id']
        self._config['next_id'] += 1
        self._save()
        return str(nid)

    def get_record_location(self, record_id: str) -> Optional[Dict[str, int]]:
        return self._config['records'].get(str(record_id))

    def get_all_record_locations(self) -> Dict[str, Dict[str, int]]:
        # return a shallow copy to avoid external mutation
        return dict(self._config['records'])

    def get_deleted_locations(self) -> List[Dict[str, int]]:
        # return a shallow copy of the tombstone list
        return list(self._config['deleted_records'])

    def add_record(self, record_id: str, offset: int, size: int) -> None:
        self._config['records'][str(record_id)] = {
            'offset': offset,
            'size': size
        }
        self._save()

    def delete_record(self, record_id: str) -> None:
        rid = str(record_id)
        if rid in self._config['records']:
            del self._config['records'][rid]
            self._save()

    def mark_deleted(self, record_id: str) -> None:
        rid = str(record_id)
        loc = self._config['records'].get(rid)
        if loc:
            # append tombstone
            self._config['deleted_records'].append({
                'offset': loc['offset'],
                'size': loc['size']
            })
            self._save()