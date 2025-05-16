import argparse
import asyncio
import json
import sys
import unittest
from io import StringIO
from unittest.mock import AsyncMock, patch, Mock

import numpy as np

from vecraft_cli.vecraft_cli import (
    parse_vector,
    get_parser,
    execute_command,
    _handle_list_collections,
    _handle_create_collection,
    _handle_insert,
    _handle_get,
    _handle_delete,
    _handle_search,
    main,
    _has_direct_command_args,
    _is_exit_command,
    _is_help_command,
    _handle_direct,
    _handle_interactive,
)


class TestVecraftCLI(unittest.TestCase):

    def test_parse_vector_comma_separated(self):
        result = parse_vector("0.1,0.2,0.3")
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_parse_vector_json_format(self):
        result = parse_vector("[0.1, 0.2, 0.3]")
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_parse_vector_with_spaces(self):
        result = parse_vector(" 0.1 , 0.2 , 0.3 ")
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_parse_vector_invalid_format(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_vector("invalid,vector,format")

    def test_get_parser(self):
        parser = get_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_has_direct_command_args(self):
        with patch.object(sys, 'argv', ['vecraft']):
            self.assertFalse(_has_direct_command_args())

        with patch.object(sys, 'argv', ['vecraft', 'list-collections']):
            self.assertTrue(_has_direct_command_args())

        with patch.object(sys, 'argv', ['vecraft', '--root', '/path']):
            self.assertFalse(_has_direct_command_args())

    def test_is_exit_command(self):
        self.assertTrue(_is_exit_command("exit"))
        self.assertTrue(_is_exit_command("quit"))
        self.assertTrue(_is_exit_command("EXIT"))
        self.assertFalse(_is_exit_command("help"))

    def test_is_help_command(self):
        self.assertTrue(_is_help_command("help"))
        self.assertTrue(_is_help_command("?"))
        self.assertTrue(_is_help_command("h"))
        self.assertFalse(_is_help_command("exit"))


class TestCommandHandlers(unittest.TestCase):

    def setUp(self):
        # Use AsyncMock for async client methods
        self.client = AsyncMock()

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_list_collections(self, mock_stdout):
        # Mock async return
        collection1 = {'name': 'collection1'}
        collection2 = {'name': 'collection2'}
        self.client.list_collections.return_value = [collection1, collection2]

        asyncio.run(_handle_list_collections(self.client))

        output = mock_stdout.getvalue().strip()
        expected = str(["collection1", "collection2"])
        self.assertEqual(output, expected)

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_create_collection(self, mock_stdout):
        args = argparse.Namespace(name="test_collection", dim=128, type="float32")
        # Mock async create_collection
        self.client.create_collection.return_value = None

        asyncio.run(_handle_create_collection(self.client, args))

        self.client.create_collection.assert_awaited_once()
        output = mock_stdout.getvalue()
        self.assertIn("Created collection 'test_collection'", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_insert(self, mock_stdout):
        args = argparse.Namespace(
            id="record1",
            vector=np.array([0.1, 0.2, 0.3]),
            data='{"text": "sample data"}',
            metadata='{"category": "test"}',
            collection="test_collection"
        )
        # Mock async insert
        self.client.insert.return_value = "record1"

        asyncio.run(_handle_insert(self.client, args))

        self.client.insert.assert_awaited_once()
        output = mock_stdout.getvalue().strip()
        self.assertEqual(output, "record1")

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_get(self, mock_stdout):
        args = argparse.Namespace(collection="test_collection", id="record1")
        # Mock async get
        record = Mock()
        record.to_dict.return_value = {
            "id": "record1",
            "vector": [0.1, 0.2, 0.3],
            "original_data": {"text": "sample data"}
        }
        self.client.get.return_value = record

        asyncio.run(_handle_get(self.client, args))

        self.client.get.assert_awaited_once_with("test_collection", "record1")
        parsed_output = json.loads(mock_stdout.getvalue())
        self.assertEqual(parsed_output["id"], "record1")

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_delete(self, mock_stdout):
        args = argparse.Namespace(collection="test_collection", id="record1")
        # Mock async delete
        self.client.delete.return_value = None

        asyncio.run(_handle_delete(self.client, args))

        self.client.delete.assert_awaited_once_with("test_collection", "record1")
        output = mock_stdout.getvalue()
        self.assertIn("Deleted record 'record1'", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_search(self, mock_stdout):
        args = argparse.Namespace(
            collection="test_collection",
            vector=np.array([0.1, 0.2, 0.3]),
            k=5,
            where='{}',
            where_document='{}'
        )
        # Mock async search
        result1 = Mock()
        result1.to_dict.return_value = {"id": "record1", "score": 0.95}
        result2 = Mock()
        result2.to_dict.return_value = {"id": "record2", "score": 0.85}
        self.client.search.return_value = [result1, result2]

        asyncio.run(_handle_search(self.client, args))

        self.client.search.assert_awaited_once()
        parsed_output = json.loads(mock_stdout.getvalue())
        self.assertEqual(len(parsed_output), 2)


class TestExecuteCommand(unittest.TestCase):

    def setUp(self):
        self.client = AsyncMock()

    @patch('vecraft_cli.vecraft_cli._handle_list_collections', new_callable=AsyncMock)
    def test_execute_command_list_collections(self, mock_handler):
        args = argparse.Namespace(command="list-collections")

        result = asyncio.run(execute_command(self.client, args))

        self.assertTrue(result)
        mock_handler.assert_awaited_once_with(self.client, args)

    def test_execute_command_invalid(self):
        args = argparse.Namespace(command="invalid-command")

        result = asyncio.run(execute_command(self.client, args))

        self.assertFalse(result)

    @patch('sys.stderr', new_callable=StringIO)
    @patch('vecraft_cli.vecraft_cli._handle_insert', new_callable=AsyncMock)
    def test_execute_command_with_exception(self, mock_handler, mock_stderr):
        args = argparse.Namespace(command="insert")
        mock_handler.side_effect = Exception("Test error")

        result = asyncio.run(execute_command(self.client, args))

        self.assertTrue(result)
        self.assertIn("Error: Test error", mock_stderr.getvalue())


class TestMainFunction(unittest.TestCase):

    @patch('vecraft_cli.vecraft_cli._has_direct_command_args', return_value=True)
    @patch('vecraft_cli.vecraft_cli.get_parser')
    @patch('vecraft_cli.vecraft_cli._handle_direct')
    def test_main_direct_mode(self, mock_direct, mock_get_parser, mock_has_direct):
        parser = Mock()
        mock_get_parser.return_value = parser

        main()

        mock_direct.assert_called_once_with(parser)

    @patch('vecraft_cli.vecraft_cli._has_direct_command_args', return_value=False)
    @patch('vecraft_cli.vecraft_cli.get_parser')
    @patch('vecraft_cli.vecraft_cli._handle_interactive')
    def test_main_interactive_mode(self, mock_interactive, mock_get_parser, mock_has_direct):
        parser = Mock()
        mock_get_parser.return_value = parser

        main()

        mock_interactive.assert_called_once_with(parser)


class TestRunDirect(unittest.TestCase):

    @patch('vecraft_cli.vecraft_cli._run_with_rest', new_callable=AsyncMock)
    @patch('sys.argv', ['vecraft', 'list-collections'])
    def test_run_direct_with_command(self, mock_run_rest):
        parser = get_parser()
        _handle_direct(parser)
        mock_run_rest.assert_awaited_once()

    @patch('sys.exit')
    @patch('sys.argv', ['vecraft'])
    def test_run_direct_no_command(self, mock_exit):
        parser = get_parser()
        _handle_direct(parser)
        mock_exit.assert_called_once_with(0)


class TestRunInteractive(unittest.TestCase):

    @patch('builtins.input', side_effect=['exit'])
    @patch('sys.stdout', new_callable=StringIO)
    def test_run_interactive_exit(self, mock_stdout, mock_input):
        parser = get_parser()
        _handle_interactive(parser)
        out = mock_stdout.getvalue()
        self.assertIn("Entering interactive mode", out)
        self.assertIn("Bye!", out)

    @patch('builtins.input', side_effect=['list-collections', 'exit'])
    @patch('vecraft_cli.vecraft_cli._run_with_rest', new_callable=AsyncMock)
    @patch('sys.stdout', new_callable=StringIO)
    def test_run_interactive_command(self, mock_stdout, mock_run_rest, mock_input):
        parser = get_parser()
        _handle_interactive(parser)
        mock_run_rest.assert_awaited_once()
        self.assertIn("Entering interactive mode", mock_stdout.getvalue())


if __name__ == '__main__':
    unittest.main()