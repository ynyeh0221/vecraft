
import typer

app = typer.Typer(help="Command-line interface for vecraft")

@app.command()
def insert(collection: str, file: str):
    """Insert vectors from FILE into COLLECTION."""
    # TODO: load data and call VectorDB.insert
    pass

@app.command()
def search(collection: str, query: str, k: int = 5):
    """Search COLLECTION with QUERY raw input, returning top-k."""
    # TODO: encode query and call VectorDB.search
    pass

if __name__ == "__main__":
    app()