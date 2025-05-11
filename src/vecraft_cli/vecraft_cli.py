import argparse
import json
import os
import shlex
import sys

import numpy as np

from src.vecraft_db.client.vecraft_client import VecraftClient
from src.vecraft_db.core.data_model.data_packet import DataPacket
from src.vecraft_db.core.data_model.index_packets import CollectionSchema
from src.vecraft_db.core.data_model.query_packet import QueryPacket


def parse_vector(vector_str):
    """
    Parse a comma-separated list of floats or JSON-style list into a NumPy array.
    Examples:
      "0.1,0.2,0.3"
      "[0.1, 0.2, 0.3]"
    """
    try:
        s = vector_str.strip()
        # JSON list syntax
        if s.startswith("[") and s.endswith("]"):
            # parse with json.loads to allow spaces
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
    default_root = os.environ.get("VCRAFT_ROOT", os.getcwd())
    parser.add_argument(
        "--root",
        default=default_root,
        help="Root directory for Vecraft data and catalog"
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
    c_insert.add_argument("vector", type=parse_vector, help="Vector as 0.1,0.2,0.3 or [0.1, 0.2, 0.3]")
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

    c_tsne = subparsers.add_parser("tsne-plot", help="Generate t-SNE plot for a collection")
    c_tsne.add_argument("collection", help=collection_name_string)
    c_tsne.add_argument("--record-ids", nargs="+", help="Specific record IDs to visualize (default: all)")
    c_tsne.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity parameter (default: 30)")
    c_tsne.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    c_tsne.add_argument("--outfile", default="tsne.png", help="Output file path (default: tsne.png)")

    return parser


class DataPacketType:
    pass


# Map command names to handler functions
_COMMAND_HANDLERS = {
    "list-collections":    "_handle_list_collections",
    "create-collection":   "_handle_create_collection",
    "insert":              "_handle_insert",
    "get":                 "_handle_get",
    "delete":              "_handle_delete",
    "search":              "_handle_search",
    "tsne-plot":           "_handle_tsne_plot",
}

def execute_command(client, args):
    handler_name = _COMMAND_HANDLERS.get(args.command)
    if not handler_name:
        return False

    try:
        globals()[handler_name](client, args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    return True


def _handle_list_collections(client, args=None): # NOSONAR
    names = [col.name for col in client.list_collections()]
    print(json.dumps(names, indent=2))


def _handle_create_collection(client, args):
    schema = CollectionSchema(name=args.name, dim=args.dim, vector_type=args.type)
    client.create_collection(schema)
    print(f"Created collection '{args.name}'")


def _handle_insert(client, args):
    data = json.loads(args.data)
    meta = json.loads(args.metadata)
    packet = DataPacket.create_record(
        record_id=args.id,
        vector=args.vector,
        original_data=data,
        metadata=meta or {},
    )
    rec_id = client.insert(collection=args.collection, packet=packet)
    print(rec_id)


def _handle_get(client, args):
    rec = client.get(args.collection, args.id)
    def _default(o):
        return o.tolist() if isinstance(o, np.ndarray) else o
    print(json.dumps(rec.to_dict(), indent=2, default=_default))


def _handle_delete(client, args):
    client.delete(args.collection, args.id)
    print(f"Deleted record '{args.id}'")


def _handle_search(client, args):
    where = json.loads(args.where) or {}
    where_doc = json.loads(args.where_document) or {}
    packet = QueryPacket(args.vector, k=args.k, where=where, where_document=where_doc)
    results = client.search(collection=args.collection, packet=packet)
    def _default(o):
        return o.tolist() if isinstance(o, np.ndarray) else o
    print(json.dumps([r.to_dict() for r in results], indent=2, default=_default))


def _handle_tsne_plot(client, args):
    path = client.generate_tsne_plot(
        collection=args.collection,
        record_ids=args.record_ids,
        perplexity=args.perplexity,
        random_state=args.random_state,
        outfile=args.outfile,
    )
    print(f"Generated t-SNE plot for collection '{args.collection}'")
    print(f"Plot saved to: {path}")


def main():
    parser = get_parser()
    if _has_direct_command_args():
        _run_direct(parser)
    else:
        _run_interactive(parser)

def _has_direct_command_args() -> bool:
    # non-flag first argument signals a one-off command invocation
    return len(sys.argv) > 1 and not sys.argv[1].startswith('-')

def _run_direct(parser):
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    client = VecraftClient(root=args.root)
    execute_command(client, args)

def _run_interactive(parser):
    print("Entering interactive mode (type 'help' for commands, 'exit' to quit)")
    client = VecraftClient(root=os.environ.get("VCRAFT_ROOT", os.getcwd()))

    while True:
        try:
            line = input("vecraft> ")
        except EOFError:
            break

        line = line.strip()
        if not line or _is_exit_command(line):
            break

        if _is_help_command(line):
            parser.print_help()
            continue

        try:
            parts = shlex.split(line)
            args = parser.parse_args(parts)
            if not args.command:
                print("No command given. Type 'help' to see available commands.")
                continue
            execute_command(client, args)

        except SystemExit as e:
            # Reraise help or explicit exits so the app actually stops;
            # otherwise treat as a parse error and continue.
            if e.code == 0:
                raise
            print(f"Error: {e}", file=sys.stderr)

    print("Bye!")

def _is_exit_command(line: str) -> bool:
    return line.lower() in ("exit", "quit")

def _is_help_command(line: str) -> bool:
    return line.lower() in ("help", "?", "h")


def entry_point():
    main()


if __name__ == "__main__":
    main()
