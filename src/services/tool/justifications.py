import json, glob, os

def load_justifications(gene):
    """Load pre-generated LLM justifications for a gene, keyed by cellosaurus_id."""
    path = f"outputs/agentic_results_{gene}.json"
    if not os.path.exists(path):
        return {}, ""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, ""
    just = {}
    for r in data.get("results", []):
        cid = r.get("cellosaurus_id")
        if cid:
            just[cid] = r.get("justification", "")
    return just, data.get("comparative_summary", "")

