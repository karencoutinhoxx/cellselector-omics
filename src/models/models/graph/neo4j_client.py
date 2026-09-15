import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        uri      = os.getenv("NEO4J_URI")
        user     = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def run_query(query: str, parameters: dict | None = None) -> list[dict]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]

