import time
import warnings

import requests

_RATE_SLEEP = 0.35
_KEGG_BASE = "https://rest.kegg.jp"

# All human KEGG pathway names, fetched once per process: {hsa04010: "Cell cycle"}
_ALL_PATHWAY_NAMES: dict[str, str] | None = None

# All human KEGG gene ID -> primary symbol, fetched once per process:
# {"hsa:1956": "EGFR", ...}. One bulk request replaces one request per gene,
# which is what makes resolving a whole pathway's membership (50-300+ genes)
# tractable instead of minutes of rate-limited per-gene lookups.
_ALL_GENE_SYMBOLS: dict[str, str] | None = None

# Gene symbols per pathway, fetched once per process: {hsa04012: ["EGFR", ...]}
_PATHWAY_GENES_CACHE: dict[str, list[str]] = {}

# KEGG's own CLASS taxonomy per pathway, fetched once per process (and
# shared across every gene that happens to share a pathway — there are only
# ~350 human KEGG pathways total, so this cache saturates quickly even
# across a 30-gene validation set): {hsa04010: "Environmental Information
# Processing; Signal transduction"}
_PATHWAY_CLASS_CACHE: dict[str, str] = {}

# Every comma-separated alias for every human KEGG gene ID, fetched once
# from the SAME bulk request as _ensure_gene_symbols (no extra KEGG calls):
# {"hsa:8503": ["P3R3URF-PIK3R3", "PIK3R3"]}. get_genes_in_pathway() only
# ever returns the FIRST alias (_ensure_gene_symbols' "primary symbol") —
# which for some genes is a readthrough-transcript or otherwise unusual
# HGNC symbol (e.g. "P3R3URF-PIK3R3") that doesn't exist in the expression
# parquet's gene_symbol column, even though a same-gene alias (PIK3R3) does.
# resolve_gene_symbol() below uses this to recover those.
_ALL_GENE_ALIASES: dict[str, list[str]] | None = None


def _kegg_get(endpoint: str) -> str | None:
    url = f"{_KEGG_BASE}/{endpoint}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        return resp.text
    except requests.exceptions.RequestException as exc:
        warnings.warn(f"[pathways] KEGG request failed ({url}): {exc}")
        return None


def _ensure_pathway_names() -> dict[str, str]:
    """Fetch and cache all human KEGG pathway names (one HTTP request per process)."""
    global _ALL_PATHWAY_NAMES
    if _ALL_PATHWAY_NAMES is not None:
        return _ALL_PATHWAY_NAMES

    text = _kegg_get("list/pathway/hsa")
    names: dict[str, str] = {}
    if text:
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                pid = parts[0].strip()  # e.g. "hsa04010"
                raw = parts[1].strip()
                names[pid] = raw.split(" - ")[0].strip()  # drop " - Homo sapiens (human)"
    _ALL_PATHWAY_NAMES = names
    return names


def _find_kegg_gene_id(gene: str) -> str | None:
    """Return the KEGG human gene ID (e.g. 'hsa:7157') for a gene symbol."""
    text = _kegg_get(f"find/hsa/{gene}")
    if not text:
        return None
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        kegg_id = parts[0].strip()
        # desc format: "TP53, tumor protein p53; K04451"
        symbols = [s.strip().upper() for s in parts[1].split(";")[0].split(",")]
        if gene.upper() in symbols:
            return kegg_id
    # Fall back to first hit
    first_line = text.splitlines()[0].split("\t")[0].strip()
    return first_line if first_line else None


def _get_pathway_class(pathway_id: str) -> str:
    """
    KEGG's own CLASS taxonomy for a pathway, e.g. "Environmental Information
    Processing; Signal transduction" or "Human Diseases; Cancer: overview".
    One request per pathway (cached — see _PATHWAY_CLASS_CACHE). Returns ""
    if KEGG is unreachable or the pathway has no CLASS line (rare, but not
    every KEGG entry has one).
    """
    if pathway_id in _PATHWAY_CLASS_CACHE:
        return _PATHWAY_CLASS_CACHE[pathway_id]

    time.sleep(_RATE_SLEEP)
    text = _kegg_get(f"get/{pathway_id}")
    cls = ""
    if text:
        for line in text.splitlines():
            if line.startswith("CLASS"):
                cls = line.replace("CLASS", "", 1).strip()
                break
    _PATHWAY_CLASS_CACHE[pathway_id] = cls
    return cls


def _is_relevant_pathway(pathway_id: str) -> bool:
    """
    Filters out KEGG's generic disease-overview / drug-resistance /
    infectious-disease / neurodegenerative-disease categories and viral
    information-processing pathways — these are real, correctly-curated
    KEGG entries (a gene showing up under "Pathways in cancer" or "Virion"
    isn't a data error), but they're not mechanistic signaling pathways
    relevant to "what cell line models this gene's biology", and diluted
    the naive pathway_activity_score with irrelevant gene sets (see the
    investigation that found this: BRCA1 lost 43% of its pathway list to
    this noise, ESR1 lost 50%).

    Deliberately NOT an ID-range heuristic (hsa04xxx = "signaling", etc.) —
    tried that first and it wrongly excludes BRCA1's two most relevant
    pathways (Homologous recombination / Fanconi anemia, both classed under
    "Genetic Information Processing; Replication and repair", not
    "Environmental Information Processing"). CLASS-based filtering is the
    real KEGG taxonomy, not a guess from the ID.
    """
    cls = _get_pathway_class(pathway_id)
    if not cls:
        return True  # no CLASS data to filter on — keep rather than guess
    top = cls.split(";")[0].strip()
    if top == "Human Diseases":
        return False
    if "virus" in cls.lower() or "viral" in cls.lower():
        return False
    return True


def get_kegg_pathways(gene: str) -> list[dict]:
    """
    Return KEGG pathways for a human gene symbol, filtered to mechanistic
    pathways (see _is_relevant_pathway).

    Each dict: { id, name, url }
    Returns [] if KEGG is unreachable or the gene is not found.
    """
    name_map = _ensure_pathway_names()

    time.sleep(_RATE_SLEEP)
    kegg_id = _find_kegg_gene_id(gene)
    if not kegg_id:
        warnings.warn(f"[pathways] Gene not found in KEGG: {gene}")
        return []

    time.sleep(_RATE_SLEEP)
    pathway_text = _kegg_get(f"link/pathway/{kegg_id}")
    if not pathway_text:
        return []

    # Each line: "hsa:7157\tpath:hsa04110"
    # NOTE: fetch the TRUE full list here, THEN filter by relevance below —
    # truncating before filtering (the old `[:20]`) is what compounded the
    # noise problem: a gene's most relevant pathways aren't guaranteed to be
    # first in KEGG's link-listing order, so capping first could silently
    # drop a real signaling pathway while keeping an irrelevant one that
    # happened to sort earlier.
    all_pathways: list[dict] = []
    for line in pathway_text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        pid_raw = parts[1].strip()  # "path:hsa04110"
        if not pid_raw.startswith("path:"):
            continue
        pid = pid_raw[5:]  # "hsa04110"
        all_pathways.append({
            "id":   pid,
            "name": name_map.get(pid, pid),
            "url":  f"https://www.kegg.jp/pathway/{pid}",
        })

    return [p for p in all_pathways if _is_relevant_pathway(p["id"])]


def _ensure_gene_symbols() -> dict[str, str]:
    """
    Fetch and cache KEGG ID -> primary symbol for every human gene, AND
    every gene's full alias list (_ALL_GENE_ALIASES) — one bulk request
    serves both, so resolve_gene_symbol()'s fallback costs nothing extra.
    """
    global _ALL_GENE_SYMBOLS, _ALL_GENE_ALIASES
    if _ALL_GENE_SYMBOLS is not None:
        return _ALL_GENE_SYMBOLS

    text = _kegg_get("list/hsa")
    symbols: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    if text:
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            kid  = parts[0].strip()          # "hsa:1956"
            desc = parts[3]                  # "EGFR, ERBB, ...; epidermal growth factor receptor..."
            alias_list = [a.strip() for a in desc.split(";")[0].split(",") if a.strip()]
            if not alias_list:
                continue
            symbols[kid] = alias_list[0]
            aliases[kid] = alias_list
    _ALL_GENE_SYMBOLS = symbols
    _ALL_GENE_ALIASES = aliases
    return symbols


def resolve_gene_symbol(symbol: str, valid_symbols: set[str]) -> str | None:
    """
    Resolve a KEGG "primary symbol" against an external gene-symbol universe
    (e.g. the expression parquet's gene_symbol column), falling back to
    same-gene aliases when the primary symbol itself isn't present there.

    Returns `symbol` unchanged if it's already in `valid_symbols` (the
    common case — no extra work); otherwise searches every KEGG gene's full
    alias list for one containing `symbol`, and returns the first of that
    gene's OTHER aliases that IS in `valid_symbols`. Returns None if no
    alias resolves.
    """
    if symbol in valid_symbols:
        return symbol

    _ensure_gene_symbols()  # populates _ALL_GENE_ALIASES as a side effect
    if not _ALL_GENE_ALIASES:
        return None

    for alias_list in _ALL_GENE_ALIASES.values():
        if symbol in alias_list:
            for alt in alias_list:
                if alt != symbol and alt in valid_symbols:
                    return alt
            break  # found the gene but no alias resolves — no point scanning further
    return None


def get_genes_in_pathway(pathway_id: str) -> list[str]:
    """
    Given a KEGG pathway ID (e.g. 'hsa04012'), return ALL gene symbols that
    are members of that pathway — no truncation.

    Reverse of get_kegg_pathways(): that goes gene -> pathways, this goes
    pathway -> genes. Together they let a caller do a 2-hop
    gene -> pathway -> gene graph traversal.

    Previously capped at max_genes=200 as a "safety cap" — but several real
    KEGG pathways exceed that (PI3K-Akt signaling: 350+, MAPK signaling:
    200+), so the cap was silently truncating membership to an arbitrary
    KEGG-listing-order prefix, not a representative sample. That biases any
    consumer that cares about a pathway's true composition (ssGSEA's
    enrichment statistic depends on true gene set size/composition; even the
    naive presence/absence scorer was checking against an incomplete set).
    No cap is reintroduced here: every caller of this function only reaches
    it via get_kegg_pathways()'s relevance filter first (see
    _is_relevant_pathway), so by the time a pathway_id gets here it's
    already been confirmed mechanistically relevant — gene ID -> symbol
    resolution is one cached bulk lookup (_ensure_gene_symbols), so
    resolving a full pathway (up to a few hundred genes) is still one
    additional KEGG request, not one request per gene, regardless of size.
    """
    symbol_map = _ensure_gene_symbols()

    text = _kegg_get(f"link/hsa/{pathway_id}")
    if not text:
        return []

    # Each line: "path:hsa04012\thsa:1956"
    kegg_gene_ids: list[str] = []
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            kegg_gene_ids.append(parts[1].strip())

    gene_symbols: list[str] = []
    for kid in kegg_gene_ids:
        symbol = symbol_map.get(kid)
        if symbol:
            gene_symbols.append(symbol)

    return gene_symbols


def get_cached_pathway_genes(pathway_id: str) -> list[str]:
    if pathway_id not in _PATHWAY_GENES_CACHE:
        _PATHWAY_GENES_CACHE[pathway_id] = get_genes_in_pathway(pathway_id)
    return _PATHWAY_GENES_CACHE[pathway_id]
