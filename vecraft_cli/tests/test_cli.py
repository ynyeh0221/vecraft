import argparse
import json
import sys
import unittest
from io import StringIO
from unittest.mock import Mock, patch

import numpy as np

from vecraft_cli.vecraft_cli import _run_interactive, _run_direct
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
    _handle_tsne_plot,
    main,
    _has_direct_command_args,
    _is_exit_command,
    _is_help_command
)


class TestVecraftCLI(unittest.TestCase):

    def test_parse_vector_comma_separated(self):
        """Test parsing comma-separated vector string"""
        result = parse_vector("0.1,0.2,0.3")
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_parse_vector_json_format(self):
        """Test parsing JSON-formatted vector string"""
        result = parse_vector("[0.1, 0.2, 0.3]")
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_parse_vector_with_spaces(self):
        """Test parsing vector string with spaces"""
        result = parse_vector(" 0.1 , 0.2 , 0.3 ")
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_parse_vector_invalid_format(self):
        """Test parsing invalid vector format raises error"""
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_vector("invalid,vector,format")

    def test_get_parser(self):
        """Test that get_parser returns an ArgumentParser"""
        parser = get_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_has_direct_command_args(self):
        """Test _has_direct_command_args function"""
        with patch.object(sys, 'argv', ['vecraft']):
            self.assertFalse(_has_direct_command_args())

        with patch.object(sys, 'argv', ['vecraft', 'list-collections']):
            self.assertTrue(_has_direct_command_args())

        with patch.object(sys, 'argv', ['vecraft', '--root', '/path']):
            self.assertFalse(_has_direct_command_args())

    def test_is_exit_command(self):
        """Test _is_exit_command function"""
        self.assertTrue(_is_exit_command("exit"))
        self.assertTrue(_is_exit_command("quit"))
        self.assertTrue(_is_exit_command("EXIT"))
        self.assertFalse(_is_exit_command("help"))

    def test_is_help_command(self):
        """Test _is_help_command function"""
        self.assertTrue(_is_help_command("help"))
        self.assertTrue(_is_help_command("?"))
        self.assertTrue(_is_help_command("h"))
        self.assertFalse(_is_help_command("exit"))


class TestCommandHandlers(unittest.TestCase):

    def setUp(self):
        self.client = Mock()

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_list_collections(self, mock_stdout):
        """Test listing collections"""
        # Mock collections
        collection1 = Mock()
        collection1.name = "collection1"
        collection2 = Mock()
        collection2.name = "collection2"

        self.client.list_collections.return_value = [collection1, collection2]

        _handle_list_collections(self.client)

        output = mock_stdout.getvalue()
        expected = json.dumps(["collection1", "collection2"], indent=2)
        self.assertEqual(output.strip(), expected)

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_create_collection(self, mock_stdout):
        """Test creating a collection"""
        args = Mock()
        args.name = "test_collection"
        args.dim = 128
        args.type = "float32"

        _handle_create_collection(self.client, args)

        self.client.create_collection.assert_called_once()
        output = mock_stdout.getvalue()
        self.assertIn("Created collection 'test_collection'", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_insert(self, mock_stdout):
        """Test inserting a record"""
        args = Mock()
        args.id = "record1"
        args.vector = np.array([0.1, 0.2, 0.3])
        args.data = '{"text": "sample data"}'
        args.metadata = '{"category": "test"}'
        args.collection = "test_collection"

        self.client.insert.return_value = "record1"

        _handle_insert(self.client, args)

        self.client.insert.assert_called_once()
        output = mock_stdout.getvalue()
        self.assertEqual(output.strip(), "record1")

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_get(self, mock_stdout):
        """Test getting a record"""
        args = Mock()
        args.collection = "test_collection"
        args.id = "record1"

        # Mock record with to_dict method
        record = Mock()
        record.to_dict.return_value = {
            "id": "record1",
            "vector": [0.1, 0.2, 0.3],
            "original_data": {"text": "sample data"}
        }

        self.client.get.return_value = record

        _handle_get(self.client, args)

        self.client.get.assert_called_once_with("test_collection", "record1")
        output = mock_stdout.getvalue()

        # Check that output is valid JSON
        parsed_output = json.loads(output)
        self.assertEqual(parsed_output["id"], "record1")

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_delete(self, mock_stdout):
        """Test deleting a record"""
        args = Mock()
        args.collection = "test_collection"
        args.id = "record1"

        _handle_delete(self.client, args)

        self.client.delete.assert_called_once_with("test_collection", "record1")
        output = mock_stdout.getvalue()
        self.assertIn("Deleted record 'record1'", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_search(self, mock_stdout):
        """Test searching records"""
        args = Mock()
        args.collection = "test_collection"
        args.vector = np.array([0.1, 0.2, 0.3])
        args.k = 5
        args.where = '{}'
        args.where_document = '{}'

        # Mock search results
        result1 = Mock()
        result1.to_dict.return_value = {"id": "record1", "score": 0.95}
        result2 = Mock()
        result2.to_dict.return_value = {"id": "record2", "score": 0.85}

        self.client.search.return_value = [result1, result2]

        _handle_search(self.client, args)

        self.client.search.assert_called_once()
        output = mock_stdout.getvalue()

        # Check that output is valid JSON
        parsed_output = json.loads(output)
        self.assertEqual(len(parsed_output), 2)

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_tsne_plot(self, mock_stdout):
        """Test generating t-SNE plot"""
        args = Mock()
        args.collection = "test_collection"
        args.record_ids = None
        args.perplexity = 30
        args.random_state = 42
        args.outfile = "tsne.png"

        self.client.generate_tsne_plot.return_value = "/path/to/tsne.png"

        _handle_tsne_plot(self.client, args)

        self.client.generate_tsne_plot.assert_called_once_with(
            collection="test_collection",
            record_ids=None,
            perplexity=30,
            random_state=42,
            outfile="tsne.png"
        )

        output = mock_stdout.getvalue()
        self.assertIn("Generated t-SNE plot", output)
        self.assertIn("/path/to/tsne.png", output)


class TestExecuteCommand(unittest.TestCase):

    def setUp(self):
        self.client = Mock()

    @patch('vecraft_db.cli.vecraft_cli._handle_list_collections')
    def test_execute_command_list_collections(self, mock_handler):
        """Test execute_command with list-collections"""
        args = Mock()
        args.command = "list-collections"

        result = execute_command(self.client, args)

        self.assertTrue(result)
        mock_handler.assert_called_once_with(self.client, args)

    def test_execute_command_invalid(self):
        """Test execute_command with invalid command"""
        args = Mock()
        args.command = "invalid-command"

        result = execute_command(self.client, args)

        self.assertFalse(result)

    @patch('vecraft_db.cli.vecraft_cli._handle_insert')
    @patch('sys.stderr', new_callable=StringIO)
    def test_execute_command_with_exception(self, mock_stderr, mock_handler):
        """Test execute_command handles exceptions"""
        args = Mock()
        args.command = "insert"

        mock_handler.side_effect = Exception("Test error")

        result = execute_command(self.client, args)

        self.assertTrue(result)
        error_output = mock_stderr.getvalue()
        self.assertIn("Error: Test error", error_output)


class TestMainFunction(unittest.TestCase):

    @patch('vecraft_db.cli.vecraft_cli._run_direct')
    @patch('vecraft_db.cli.vecraft_cli._has_direct_command_args')
    @patch('vecraft_db.cli.vecraft_cli.get_parser')
    def test_main_direct_mode(self, mock_get_parser, mock_has_direct, mock_run_direct):
        """Test main function in direct mode"""
        mock_has_direct.return_value = True
        parser = Mock()
        mock_get_parser.return_value = parser

        main()

        mock_run_direct.assert_called_once_with(parser)

    @patch('vecraft_db.cli.vecraft_cli._run_interactive')
    @patch('vecraft_db.cli.vecraft_cli._has_direct_command_args')
    @patch('vecraft_db.cli.vecraft_cli.get_parser')
    def test_main_interactive_mode(self, mock_get_parser, mock_has_direct, mock_run_interactive):
        """Test main function in interactive mode"""
        mock_has_direct.return_value = False
        parser = Mock()
        mock_get_parser.return_value = parser

        main()

        mock_run_interactive.assert_called_once_with(parser)


class TestRunDirect(unittest.TestCase):

    @patch('vecraft_db.cli.vecraft_cli.VecraftClient')
    @patch('vecraft_db.cli.vecraft_cli.execute_command')
    @patch('sys.argv', ['vecraft', 'list-collections'])
    def test_run_direct_with_command(self, mock_execute, mock_client):
        """Test _run_direct with a valid command"""
        parser = get_parser()
        client_instance = Mock()
        mock_client.return_value = client_instance

        _run_direct(parser)

        mock_execute.assert_called_once()
        args = mock_execute.call_args[0][1]
        self.assertEqual(args.command, 'list-collections')

    @patch('sys.exit')
    @patch('sys.argv', ['vecraft'])
    def test_run_direct_no_command(self, mock_exit):
        """Test _run_direct with no command"""

        parser = get_parser()

        _run_direct(parser)

        mock_exit.assert_called_once_with(0)


class TestRunInteractive(unittest.TestCase):

    @patch('vecraft_db.cli.vecraft_cli.VecraftClient')
    @patch('builtins.input', side_effect=['exit'])
    @patch('sys.stdout', new_callable=StringIO)
    def test_run_interactive_exit(self, mock_stdout, mock_input, mock_client):
        """Test _run_interactive with exit command"""

        parser = get_parser()

        _run_interactive(parser)

        output = mock_stdout.getvalue()
        self.assertIn("Entering interactive mode", output)
        self.assertIn("Bye!", output)

    @patch('vecraft_db.cli.vecraft_cli.VecraftClient')
    @patch('vecraft_db.cli.vecraft_cli.execute_command')
    @patch('builtins.input', side_effect=['list-collections', 'exit'])
    @patch('sys.stdout', new_callable=StringIO)
    def test_run_interactive_command(self, mock_stdout, mock_input, mock_execute, mock_client):
        """Test _run_interactive with a command followed by exit"""

        parser = get_parser()
        client_instance = Mock()
        mock_client.return_value = client_instance

        _run_interactive(parser)

        mock_execute.assert_called_once()
        args = mock_execute.call_args[0][1]
        self.assertEqual(args.command, 'list-collections')


if __name__ == '__main__':
    unittest.main()