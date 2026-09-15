"""
Shared FastAPI dependencies.

Heavy data (DataFrames, gene sets, eval stats) is loaded once during the
lifespan event in main.py and stored on app.state.  These helpers extract
it from the current request for use in endpoint handlers.
"""
from fastapi import Request
import pandas as pd


def get_master_merged(request: Request) -> pd.DataFrame:
    return request.app.state.master_merged


def get_cell_line_lookup(request: Request) -> pd.DataFrame:
    return request.app.state.cell_line_lookup


def get_gene_sets(request: Request) -> dict[str, set]:
    return request.app.state.gene_sets


def get_eval_stats(request: Request) -> dict | None:
    return request.app.state.eval_stats

