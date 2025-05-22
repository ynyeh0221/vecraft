import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import fcntl
import numpy as np

from vecraft_data_model.data_packet import DataPacket
from vecraft_db.persistence.wal_manager import WALManager


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

    @patch('fcntl.flock')
    def test_append(self, mock_flock):
        """Test appending entries to the WAL file."""
        test_entries = [
            DataPacket.create_record(
                record_id="user:123",
                original_data={"name": "John"},
                vector=np.array([0, 0, 0], dtype=np.float32),
                metadata={}
            ),
            DataPacket.create_record(
                record_id="user:123",
                original_data={"name": "John Doe"},
                vector=np.array([0, 0, 0], dtype=np.float32),
                metadata={}
            )
        ]

        # Append entries with default phase
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
            expected_entry = test_entries[i].to_dict()
            expected_entry["_phase"] = "prepare"  # Default phase
            expected_entry["_lsn"] = loaded_entry["_lsn"]
            self.assertEqual(loaded_entry, expected_entry)

    @patch('fcntl.flock')
    def test_append_with_phase(self, mock_flock):
        """Test appending entries with different phases."""
        test_packet = DataPacket.create_record(
            record_id="user:123",
            original_data={"name": "John"},
            vector=np.array([0, 0, 0], dtype=np.float32),
            metadata={}
        )

        # Append twice in prepare phase (capture second LSN)
        first_lsn = self.wal_manager.append(test_packet, phase="prepare")
        prepare_lsn = self.wal_manager.append(test_packet, phase="prepare")
        self.assertGreater(prepare_lsn, first_lsn)

        # Commit the record
        self.wal_manager.commit("user:123")

        # Read the file and verify content: two prepares + one commit
        with open(self.wal_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)

        # Check second prepare entry
        prepare_entry = json.loads(lines[1])
        expected_prepare = test_packet.to_dict()
        expected_prepare["_phase"] = "prepare"
        expected_prepare["_lsn"] = prepare_lsn
        self.assertEqual(expected_prepare, prepare_entry)

        # Check commit entry
        commit_entry = json.loads(lines[2])
        self.assertEqual(commit_entry["record_id"], "user:123")
        self.assertEqual(commit_entry["_phase"], "commit")

    @patch('fcntl.flock')
    def test_replay(self, mock_flock):
        """Test replaying entries from the WAL file."""
        test_entries = [
            DataPacket.create_record(
                record_id="user:123",
                original_data={"name": "John"},
                vector=np.array([0, 0, 0], dtype=np.float32),
                metadata={}
            ),
            DataPacket.create_record(
                record_id="user:456",
                original_data={"name": "John Doe"},
                vector=np.array([0, 0, 0], dtype=np.float32),
                metadata={}
            )
        ]

        # Append entries with commits
        for entry in test_entries:
            self.wal_manager.append(entry)
            self.wal_manager.commit(entry.record_id)

        # Create a mock handler
        mock_handler = Mock()

        # Replay the WAL
        count = self.wal_manager.replay(mock_handler)

        # Verify only committed entries were replayed
        self.assertEqual(len(test_entries), count)
        self.assertEqual(len(test_entries), mock_handler.call_count)

        # Verify the correct entries were passed to the handler
        for i, entry in enumerate(test_entries):
            print(i, entry)
            call_args = mock_handler.call_args_list[i][0][0]
            expected_args = entry.to_dict()
            expected_args["_phase"] = "prepare"
            expected_args["_lsn"] = call_args["_lsn"]
            self.assertEqual(expected_args, call_args)

        # Verify WAL file was deleted after replay
        self.assertFalse(self.wal_path.exists())

    @patch('fcntl.flock')
    def test_replay_uncommitted(self, mock_flock):
        """Test that uncommitted entries are not replayed."""
        test_entries = [
            DataPacket.create_record(
                record_id="user:123",
                original_data={"name": "John"},
                vector=np.array([0, 0, 0], dtype=np.float32),
                metadata={}
            ),
            DataPacket.create_record(
                record_id="user:456",
                original_data={"name": "Jane"},
                vector=np.array([0, 0, 0], dtype=np.float32),
                metadata={}
            )
        ]

        # Append first entry with commit, second without
        self.wal_manager.append(test_entries[0])
        self.wal_manager.commit(test_entries[0].record_id)
        self.wal_manager.append(test_entries[1])  # No commit

        # Create a mock handler
        mock_handler = Mock()

        # Replay the WAL
        count = self.wal_manager.replay(mock_handler)

        # Verify only committed entry was replayed
        self.assertEqual(1, count)
        self.assertEqual(1, mock_handler.call_count)

        # Verify the correct entry was passed to handler
        call_args = mock_handler.call_args_list[0][0][0]
        expected_args = test_entries[0].to_dict()
        expected_args["_phase"] = "prepare"
        expected_args["_lsn"] = call_args["_lsn"]
        self.assertEqual(expected_args, call_args)

    def test_clear(self):
        """Test clearing the WAL file."""
        # Append an entry to create the file
        test_packet = DataPacket.create_record(
            record_id="test",
            original_data={},
            vector=np.array([0, 0, 0], dtype=np.float32),
            metadata={}
        )
        with patch('fcntl.flock'):
            self.wal_manager.append(test_packet)
        self.assertTrue(self.wal_path.exists())

        # Clear the WAL
        self.wal_manager.clear()

        # Verify file was deleted
        self.assertFalse(self.wal_path.exists())

    def test_replay_nonexistent_file(self):
        """Test replaying a non-existent WAL file."""
        # Ensure a file doesn't exist
        if self.wal_path.exists():
            self.wal_path.unlink()

        mock_handler = Mock()

        # Replay should not raise an exception
        count = self.wal_manager.replay(mock_handler)

        # Handler should not have been called
        mock_handler.assert_not_called()
        self.assertEqual(0, count)

    def test_clear_nonexistent_file(self):
        """Test clearing a non-existent WAL file."""
        # Ensure file doesn't exist
        if self.wal_path.exists():
            self.wal_path.unlink()

        # Clear should not raise an exception
        self.wal_manager.clear()

    @patch('builtins.open')
    @patch('os.fsync')
    @patch('fcntl.flock')
    def test_append_flushes_and_syncs(self, mock_flock, mock_fsync, mock_open):
        """Test that append flushes and syncs both the LSN‐meta and WAL files."""
        # Prepare a fake file object
        mock_file = MagicMock()
        mock_file.fileno.return_value = 42
        mock_open.return_value.__enter__.return_value = mock_file

        pkt = DataPacket.create_record(
            record_id="test",
            original_data={},
            vector=np.array([0, 0, 0], dtype=np.float32),
            metadata={}
        )
        # Invoke append, which writes both to the .lsn.meta and the WAL itself
        self.wal_manager.append(pkt)

        # We expect at least one flush per file → total flush calls ≥ 2
        self.assertGreaterEqual(mock_file.flush.call_count, 2,
                                f"Expected ≥2 flush() calls, got {mock_file.flush.call_count}")

        # We expect at least one fsync per file → total fsync calls ≥ 2
        self.assertGreaterEqual(mock_fsync.call_count, 2,
                                f"Expected ≥2 fsync() calls, got {mock_fsync.call_count}")
        for args, _ in mock_fsync.call_args_list:
            self.assertEqual(args[0], 42)

        # We expect at least lock/unlock per file → total flock calls ≥ 4
        self.assertGreaterEqual(mock_flock.call_count, 4,
                                f"Expected ≥4 flock() calls, got {mock_flock.call_count}")
        seen_modes = {call.args[1] for call in mock_flock.call_args_list}
        self.assertIn(fcntl.LOCK_EX, seen_modes)
        self.assertIn(fcntl.LOCK_UN, seen_modes)

    @patch('fcntl.flock')
    def test_replay_with_corrupted_entry(self, mock_flock):
        """Test replaying with corrupted WAL entry."""
        # Write valid entry followed by corrupted entry
        with open(self.wal_path, 'w', encoding='utf-8') as f:
            # Valid prepare entry
            test_packet = DataPacket.create_record(
                record_id="user:123",
                original_data={"name": "John"},
                vector=np.array([0, 0, 0], dtype=np.float32),
                metadata={}
            )
            prepare_entry = test_packet.to_dict()
            prepare_entry["_phase"] = "prepare"
            f.write(json.dumps(prepare_entry) + "\n")

            # Corrupted line
            f.write("corrupted json\n")

        mock_handler = Mock()

        # Replay should raise exception and preserve WAL
        with self.assertRaises(Exception) as context:
            self.wal_manager.replay(mock_handler)

        self.assertIn("Corrupted WAL entry detected", str(context.exception))

        # WAL should still exist (restored)
        self.assertTrue(self.wal_path.exists())

        # Handler should not have been called
        mock_handler.assert_not_called()

    @patch('fcntl.flock')
    def test_compact_basic(self, mock_flock):
        self.wal_manager._compaction_threshold = 1
        """Test basic compaction functionality."""
        # Create test entries with different LSNs
        test_entries = []
        for i in range(5):
            entry = DataPacket.create_record(
                record_id=f"user:{i}",
                original_data={"name": f"User {i}"},
                vector=np.array([i, i, i], dtype=np.float32),
                metadata={}
            )
            test_entries.append(entry)

        # Append all entries
        lsns = []
        for entry in test_entries:
            lsn = self.wal_manager.append(entry)
            lsns.append(lsn)

        # Verify file has entries
        with open(self.wal_path, 'r', encoding='utf-8') as f:
            lines_before = f.readlines()
        self.assertEqual(len(lines_before), 5)

        # Compact with visible_lsn = 3 (should remove first 3 entries)
        visible_lsn = lsns[2]  # LSN of third entry
        entries_before, entries_after = self.wal_manager.compact(visible_lsn)

        self.assertEqual(entries_before, 5)
        self.assertEqual(entries_after, 2)

        # Verify remaining entries
        with open(self.wal_path, 'r', encoding='utf-8') as f:
            lines_after = f.readlines()
        self.assertEqual(len(lines_after), 2)

        # Check that remaining entries have LSN > visible_lsn
        for line in lines_after:
            entry = json.loads(line)
            self.assertGreater(entry["_lsn"], visible_lsn)

    @patch('fcntl.flock')
    def test_compact_all_entries_old(self, mock_flock):
        self.wal_manager._compaction_threshold = 1
        """Test compaction when all entries are old."""
        # Create test entries
        test_entries = []
        for i in range(3):
            entry = DataPacket.create_record(
                record_id=f"user:{i}",
                original_data={"name": f"User {i}"},
                vector=np.array([i, i, i], dtype=np.float32),
                metadata={}
            )
            test_entries.append(entry)

        # Append all entries
        lsns = []
        for entry in test_entries:
            lsn = self.wal_manager.append(entry)
            lsns.append(lsn)

        # Compact with visible_lsn higher than all entries
        visible_lsn = max(lsns) + 10
        entries_before, entries_after = self.wal_manager.compact(visible_lsn)

        self.assertEqual(entries_before, 3)
        self.assertEqual(entries_after, 0)

        # Verify WAL file is deleted when no entries remain
        self.assertFalse(self.wal_path.exists())

    @patch('fcntl.flock')
    def test_compact_no_entries_to_remove(self, mock_flock):
        self.wal_manager._compaction_threshold = 1
        """Test compaction when no entries need to be removed."""
        # Create test entries
        test_entries = []
        for i in range(3):
            entry = DataPacket.create_record(
                record_id=f"user:{i}",
                original_data={"name": f"User {i}"},
                vector=np.array([i, i, i], dtype=np.float32),
                metadata={}
            )
            test_entries.append(entry)

        # Append all entries
        lsns = []
        for entry in test_entries:
            lsn = self.wal_manager.append(entry)
            lsns.append(lsn)

        # Compact with visible_lsn = 0 (should remove no entries)
        visible_lsn = 0
        entries_before, entries_after = self.wal_manager.compact(visible_lsn)

        self.assertEqual(entries_before, 3)
        self.assertEqual(entries_after, 3)

        # Verify all entries remain
        with open(self.wal_path, 'r', encoding='utf-8') as f:
            lines_after = f.readlines()
        self.assertEqual(len(lines_after), 3)

    @patch('fcntl.flock')
    def test_compact_preserves_lsn_order(self, mock_flock):
        self.wal_manager._compaction_threshold = 1
        """Test that compaction preserves LSN ordering."""
        # Create entries with non-sequential record IDs to test LSN ordering
        test_data = [
            ("user:c", {"name": "Charlie"}),
            ("user:a", {"name": "Alice"}),
            ("user:b", {"name": "Bob"}),
            ("user:d", {"name": "David"}),
        ]

        lsns = []
        for record_id, data in test_data:
            entry = DataPacket.create_record(
                record_id=record_id,
                original_data=data,
                vector=np.array([1, 2, 3], dtype=np.float32),
                metadata={}
            )
            lsn = self.wal_manager.append(entry)
            lsns.append(lsn)

        # Compact to remove first entry only (use LSN of first entry as visible_lsn)
        visible_lsn = lsns[0]
        self.wal_manager.compact(visible_lsn)

        # Verify remaining entries are in LSN order
        with open(self.wal_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        prev_lsn = visible_lsn  # Start from visible_lsn
        for line in lines:
            entry = json.loads(line)
            current_lsn = entry["_lsn"]
            self.assertGreater(current_lsn, prev_lsn)
            prev_lsn = current_lsn


if __name__ == '__main__':
    unittest.main()