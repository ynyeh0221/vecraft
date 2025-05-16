#!/usr/bin/env python3
import argparse
import asyncio
import inspect
import json
import os
import shlex
import sys

import numpy as np

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.index_packets import CollectionSchema
from vecraft_data_model.query_packet import QueryPacket
from vecraft_rest_client.vecraft_rest_api_client import VecraftFastAPIClient


def parse_vector(vector_str):
    """
    Parse a comma-separated list of floats or JSON-style list into a NumPy array.
    Examples:
      "0.1,0.2,0.3"
      "[0.1, 0.2, 0.3]"
    """
    try:
        s = vector_str.strip()
        if s.startswith("[") and s.endswith("]"):
            arr = json.loads(s)
            if not isinstance(arr, list):
                raise ValueError("Not a JSON list")
            vec = np.array(arr, dtype=np.float32)
        else:
            nums = [float(x.strip()) for x in s.split(",")]
            vec = np.array(nums, dtype=np.float32)
        return vec
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid vector format: {e}")


def get_parser():
    parser = argparse.ArgumentParser(
        prog="vecraft",
        description="Vecraft CLI: manage collections and perform insert, search, get, delete operations"
    )
    default_root = os.environ.get("VCRAFT_ROOT", "http://localhost:8000")
    parser.add_argument(
        "--root",
        default=default_root,
        help="Root directory or base URL for Vecraft"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-collections", help="List all collections")

    collection_name_string = "Collection name"
    record_id_string = "Record ID"
    c_create = subparsers.add_parser("create-collection", help="Create a new collection")
    c_create.add_argument("name", help=collection_name_string)
    c_create.add_argument("dim", type=int, help="Vector dimension")
    c_create.add_argument("--type", default="float32", help="Vector type")

    c_insert = subparsers.add_parser("insert", help="Insert or update a record")
    c_insert.add_argument("collection", help=collection_name_string)
    c_insert.add_argument("id", help=record_id_string)
    c_insert.add_argument("vector", type=parse_vector,
                          help="Vector as 0.1,0.2,0.3 or [0.1, 0.2, 0.3]")
    c_insert.add_argument("--data", required=True, help="Original data as JSON string")
    c_insert.add_argument("--metadata", default="{}", help="Optional metadata as JSON string")

    c_get = subparsers.add_parser("get", help="Retrieve a record")
    c_get.add_argument("collection", help=collection_name_string)
    c_get.add_argument("id", help=record_id_string)

    c_delete = subparsers.add_parser("delete", help="Delete a record")
    c_delete.add_argument("collection", help=collection_name_string)
    c_delete.add_argument("id", help=record_id_string)

    c_search = subparsers.add_parser("search", help="K-NN search")
    c_search.add_argument("collection", help=collection_name_string)
    c_search.add_argument("vector", type=parse_vector, help="Query vector")
    c_search.add_argument("k", type=int, help="Number of neighbors to return")
    c_search.add_argument("--where", default="{}", help="Optional filter as JSON string")
    c_search.add_argument("--where-document", default="{}", help="Optional document filter as JSON string")

    return parser


# Map command names to handler functions
_COMMAND_HANDLERS = {
    "list-collections":    "_handle_list_collections",
    "create-collection":   "_handle_create_collection",
    "insert":              "_handle_insert",
    "get":                 "_handle_get",
    "delete":              "_handle_delete",
    "search":              "_handle_search",
}


async def execute_command(client, args):
    """Dispatch to handlers, awaiting coroutines."""
    handler_name = _COMMAND_HANDLERS.get(args.command)
    if not handler_name:
        return False
    handler = globals().get(handler_name)
    try:
        result = handler(client, args)
        if inspect.isawaitable(result):
            await result
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    return True


async def _handle_list_collections(client, args=None):  #NOSONAR
    collections = await client.list_collections()
    names = [col['name'] for col in collections]
    print(names)

async def _handle_create_collection(client, args):
    schema = CollectionSchema(name=args.name, dim=args.dim, vector_type=args.type)
    await client.create_collection(schema)
    print(f"Created collection '{args.name}'")


async def _handle_insert(client, args):
    data = json.loads(args.data)
    meta = json.loads(args.metadata)
    packet = DataPacket.create_record(
        record_id=args.id,
        vector=args.vector,
        original_data=data,
        metadata=meta or {},
    )
    rec_id = await client.insert(collection=args.collection, data_packet=packet)
    print(rec_id)


async def _handle_get(client, args):
    rec = await client.get(args.collection, args.id)
    def _default(o):
        return o.tolist() if isinstance(o, np.ndarray) else o
    print(json.dumps(rec.to_dict(), indent=2, default=_default))


async def _handle_delete(client, args):
    await client.delete(args.collection, args.id)
    print(f"Deleted record '{args.id}'")


async def _handle_search(client, args):
    where = json.loads(args.where) or {}
    where_doc = json.loads(args.where_document) or {}
    packet = QueryPacket(args.vector, k=args.k, where=where, where_document=where_doc)
    results = await client.search(collection=args.collection, query_packet=packet)
    def _default(o):
        return o.tolist() if isinstance(o, np.ndarray) else o
    print(json.dumps([r.to_dict() for r in results], indent=2, default=_default))


async def _run_with_rest(args):
    """Open REST client, dispatch, then close."""
    async with VecraftFastAPIClient(base_url=args.root) as client:
        await execute_command(client, args)


def _has_direct_command_args() -> bool:
    return len(sys.argv) > 1 and not sys.argv[1].startswith('-')


def main():
    parser = get_parser()
    if _has_direct_command_args():
        _handle_direct(parser)
    else:
        _handle_interactive(parser)


def _handle_direct(parser):
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    asyncio.run(_run_with_rest(args))


def _handle_interactive(parser):
    print("Entering interactive mode (type 'help' for commands, 'exit' to quit)")
    while True:
        try:
            raw = input("vecraft> ")
        except EOFError:
            break

        line = raw.strip()
        if not line or _is_exit_command(line):
            break

        if _is_help_command(line):
            parser.print_help()
            continue

        _process_interactive_line(line, parser)

    print("Bye!")


def _process_interactive_line(line, parser):
    try:
        parts = shlex.split(line)
        args = parser.parse_args(parts)
        if not args.command:
            print("No command given. Type 'help' for commands.")
            return
        asyncio.run(_run_with_rest(args))
    except SystemExit as e:
        if e.code == 0:
            raise
        print(f"Error: {e}", file=sys.stderr)


def _is_exit_command(line: str) -> bool:
    return line.lower() in ("exit", "quit")


def _is_help_command(line: str) -> bool:
    return line.lower() in ("help", "?", "h")


def entry_point():
    main()


if __name__ == "__main__":
    main()