import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.vecraft.data.checksummed_data import DataPacket
from src.vecraft.wal.wal_manager import WALManager


class TestWALManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wal_path = Path(self.temp_dir.name) / "test_wal.json"
        self.wal_manager = WALManager(self.wal_path)

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization of WALManager."""
        self.assertEqual(self.wal_manager._file, self.wal_path)
        self.assertFalse(self.wal_path.exists())

    def test_append(self):
        """Test appending entries to the WAL file."""
        test_entries = [
            DataPacket(
                type="insert",
                record_id="user:123",
                original_data={"name": "John"},
                vector=None,
                metadata={}
            ),
            DataPacket(
                type="insert",
                record_id="user:123",
                original_data={"name": "John Doe"},
                vector=None,
                metadata={}
            )
        ]

        # Append entries
        for entry in test_entries:
            self.wal_manager.append(entry)

        # Verify file exists
        self.assertTrue(self.wal_path.exists())

        # Read the file and verify content
        with open(self.wal_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), len(test_entries))

        for i, line in enumerate(lines):
            loaded_entry = json.loads(line)
            self.assertEqual(loaded_entry, test_entries[i].to_dict())

    def test_replay(self):
        """Test replaying entries from the WAL file."""
        test_entries = [
            DataPacket(
                type="insert",
                record_id="user:123",
                original_data={"name": "John"},
                vector=None,
                metadata={}
            ),
            DataPacket(
                type="insert",
                record_id="user:123",
                original_data={"name": "John Doe"},
                vector=None,
                metadata={}
            )
        ]

        # Append entries
        for entry in test_entries:
            self.wal_manager.append(entry)

        # Create a mock handler
        mock_handler = Mock()

        # Replay the WAL
        self.wal_manager.replay(mock_handler)

        # Verify handler was called with each entry
        self.assertEqual(mock_handler.call_count, len(test_entries))
        for i, entry in enumerate(test_entries):
            call_args = mock_handler.call_args_list[i][0][0]
            self.assertEqual(call_args, entry.to_dict())

        # Verify WAL file was deleted after replay
        self.assertFalse(self.wal_path.exists())

    def test_clear(self):
        """Test clearing the WAL file."""
        # Append an entry to create the file
        test_packet = DataPacket(
            type="insert",
            record_id="test",
            original_data={},
            vector=None,
            metadata={}
        )
        self.wal_manager.append(test_packet)
        self.assertTrue(self.wal_path.exists())

        # Clear the WAL
        self.wal_manager.clear()

        # Verify file was deleted
        self.assertFalse(self.wal_path.exists())

    def test_replay_nonexistent_file(self):
        """Test replaying a non-existent WAL file."""
        # Ensure file doesn't exist
        if self.wal_path.exists():
            self.wal_path.unlink()

        mock_handler = Mock()

        # Replay should not raise an exception
        self.wal_manager.replay(mock_handler)

        # Handler should not have been called
        mock_handler.assert_not_called()

    def test_clear_nonexistent_file(self):
        """Test clearing a non-existent WAL file."""
        # Ensure file doesn't exist
        if self.wal_path.exists():
            self.wal_path.unlink()

        # Clear should not raise an exception
        self.wal_manager.clear()

    @patch('builtins.open')
    @patch('os.fsync')
    def test_append_flushes_and_syncs(self, mock_fsync, mock_open):
        """Test that append flushes and syncs the file."""
        mock_file = Mock()
        mock_file.fileno.return_value = 42
        mock_open.return_value.__enter__.return_value = mock_file

        test_packet = DataPacket(
            type="insert",
            record_id="test",
            original_data={},
            vector=None,
            metadata={}
        )
        self.wal_manager.append(test_packet)

        # Verify flush and fsync were called
        mock_file.flush.assert_called_once()
        mock_fsync.assert_called_once_with(42)


if __name__ == '__main__':
    unittest.main()