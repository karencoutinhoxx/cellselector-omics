import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.graph.neo4j_client import run_query


def setup_schema():
    queries = [
        "CREATE CONSTRAINT gene_symbol IF NOT EXISTS "
        "FOR (g:Gene) REQUIRE g.symbol IS UNIQUE",

        "CREATE CONSTRAINT pathway_id IF NOT EXISTS "
        "FOR (p:Pathway) REQUIRE p.pathway_id IS UNIQUE",

        "CREATE CONSTRAINT cellline_id IF NOT EXISTS "
        "FOR (c:CellLine) REQUIRE c.cellosaurus_id IS UNIQUE",
    ]
    for q in queries:
        try:
            run_query(q)
            print(f"OK: {q[:50]}...")
        except Exception as exc:
            print(f"Skip (may already exist): {exc}")


if __name__ == "__main__":
    setup_schema()

