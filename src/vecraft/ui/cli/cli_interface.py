import argparse
import json
import os
import shlex
import sys

import numpy as np

from src.vecraft.api.vecraft_client import VecraftClient
from src.vecraft.data.index_packets import CollectionSchema, QueryPacket, DataPacket, DataPacketType


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

    c_create = subparsers.add_parser("create-collection", help="Create a new collection")
    c_create.add_argument("name", help="Collection name")
    c_create.add_argument("dim", type=int, help="Vector dimension")
    c_create.add_argument("--type", default="float32", help="Vector type")

    c_insert = subparsers.add_parser("insert", help="Insert or update a record")
    c_insert.add_argument("collection", help="Collection name")
    c_insert.add_argument("id", help="Record ID")
    c_insert.add_argument("vector", type=parse_vector, help="Vector as 0.1,0.2,0.3 or [0.1, 0.2, 0.3]")
    c_insert.add_argument("--data", required=True, help="Original data as JSON string")
    c_insert.add_argument("--metadata", default="{}", help="Optional metadata as JSON string")

    c_get = subparsers.add_parser("get", help="Retrieve a record")
    c_get.add_argument("collection", help="Collection name")
    c_get.add_argument("id", help="Record ID")

    c_delete = subparsers.add_parser("delete", help="Delete a record")
    c_delete.add_argument("collection", help="Collection name")
    c_delete.add_argument("id", help="Record ID")

    c_search = subparsers.add_parser("search", help="K-NN search")
    c_search.add_argument("collection", help="Collection name")
    c_search.add_argument("vector", type=parse_vector, help="Query vector")
    c_search.add_argument("k", type=int, help="Number of neighbors to return")
    c_search.add_argument("--where", default="{}", help="Optional filter as JSON string")
    c_search.add_argument("--where-document", default="{}", help="Optional document filter as JSON string")

    c_tsne = subparsers.add_parser("tsne-plot", help="Generate t-SNE plot for a collection")
    c_tsne.add_argument("collection", help="Collection name")
    c_tsne.add_argument("--record-ids", nargs="+", help="Specific record IDs to visualize (default: all)")
    c_tsne.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity parameter (default: 30)")
    c_tsne.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    c_tsne.add_argument("--outfile", default="tsne.png", help="Output file path (default: tsne.png)")

    return parser


def execute_command(client, args):
    try:
        if args.command == "list-collections":
            print(json.dumps([item.name for item in client.list_collections()], indent=2))
        elif args.command == "create-collection":
            client.create_collection(CollectionSchema(name=args.name, dim=args.dim, vector_type=args.type))
            print(f"Created collection '{args.name}'")
        elif args.command == "insert":
            data = json.loads(args.data)
            meta = json.loads(args.metadata)
            rec_id = client.insert(
                collection=args.collection,
                packet=DataPacket(type=DataPacketType.RECORD, record_id=args.id, vector=args.vector, original_data=data, metadata=meta or {}),
            )
            print(rec_id)
        elif args.command == "get":
            rec = client.get(args.collection, args.id)
            print(json.dumps(rec.to_dict(), indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))
        elif args.command == "delete":
            client.delete(args.collection, args.id)
            print(f"Deleted record '{args.id}'")
        elif args.command == "search":
            where = json.loads(args.where)
            where_document = json.loads(args.where_document)
            results = client.search(
                collection=args.collection,
                packet=QueryPacket(args.vector, k=args.k, where=where or {}, where_document=where_document or {}),
            )
            results_dict = [result.to_dict() for result in results]
            print(json.dumps(results_dict, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))
        elif args.command == "tsne-plot":
            # Execute the t-SNE plot generation
            plot_path = client.generate_tsne_plot(
                collection=args.collection,
                record_ids=args.record_ids,
                perplexity=args.perplexity,
                random_state=args.random_state,
                outfile=args.outfile
            )
            print(f"Generated t-SNE plot for collection '{args.collection}'")
            print(f"Plot saved to: {plot_path}")
        else:
            return False
        return True
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return True


def main():
    parser = get_parser()
    # First parse only --root to init client
    if len(sys.argv) > 1 and sys.argv[1] not in ["-h", "--help"] and not sys.argv[1].startswith("-"):
        args = parser.parse_args()
        if not args.command:
            parser.print_help()
            sys.exit(0)
        client = VecraftClient(root=args.root)
        execute_command(client, args)
    else:
        # interactive mode
        print("Entering interactive mode (type 'help' for commands, 'exit' to quit)")
        client = VecraftClient(root=os.environ.get("VCRAFT_ROOT", os.getcwd()))
        parser = get_parser()
        while True:
            try:
                line = input("vecraft> ")
            except EOFError:
                break
            if not line or line.strip().lower() in ["exit", "quit"]:
                break
            if line.strip().lower() in ["help", "?", "h"]:
                parser.print_help()
                continue
            try:
                parts = shlex.split(line)
                args = parser.parse_args(parts)
                if not args.command:
                    print("No command given. Type 'help' to see available commands.")
                    continue
                execute_command(client, args)
            except SystemExit:
                # argparse parse error
                continue
        print("Bye!")


def entry_point():
    main()


if __name__ == "__main__":
    main()
