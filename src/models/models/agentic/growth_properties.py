import requests
import time

_GROWTH_CACHE: dict = {}


def get_growth_properties(cellosaurus_id: str) -> dict | None:
    """
    Query the Cellosaurus REST API for culture-relevant metadata for a CVCL accession.

    Returns a dict with:
        category      - e.g. "Cancer cell line", "Transformed cell line"
        cell_type     - ontology-derived cell type when available (e.g. "T-cell")
        doubling_time - dict with min/max/unit when available
        karyotype     - karyotypic information comment when available

    Note: The JSON API does not expose growth mode (Adherent/Suspension) or
    cell morphology (Epithelial/Fibroblast) as structured fields — these comment
    categories are absent from the API for all tested cell lines.

    Returns None on network failure or if no useful data is found.
    """
    if cellosaurus_id in _GROWTH_CACHE:
        return _GROWTH_CACHE[cellosaurus_id]

    try:
        url = f"https://api.cellosaurus.org/cell-line/{cellosaurus_id}"
        resp = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            _GROWTH_CACHE[cellosaurus_id] = None
            return None

        data = resp.json()
        cell_line_list = (
            data.get("Cellosaurus", {})
                .get("cell-line-list", [])
        )
        if not cell_line_list:
            _GROWTH_CACHE[cellosaurus_id] = None
            return None

        cell_line = cell_line_list[0]

        # cell-type is a top-level key present on some lines (T-cell, Erythroblast, etc.)
        cell_type_obj = cell_line.get("cell-type")
        cell_type = cell_type_obj.get("value") if isinstance(cell_type_obj, dict) else None

        # doubling-time-range: {"min": "28", "max": "84", "unit": "hour"}
        dt = cell_line.get("doubling-time-range")
        if isinstance(dt, dict) and dt.get("min"):
            doubling_time = {
                "min":  dt.get("min"),
                "max":  dt.get("max"),
                "unit": dt.get("unit", "hour"),
            }
        else:
            doubling_time = None

        # Karyotypic information comment — real category name in the API
        karyotype = None
        for comment in cell_line.get("comment-list", []):
            # NOTE: real field is "value", not "comment-value"
            if comment.get("category") == "Karyotypic information":
                karyotype = comment.get("value")
                break

        growth_info = {
            "category":      cell_line.get("category"),
            "cell_type":     cell_type,
            "doubling_time": doubling_time,
            "karyotype":     karyotype,
        }

        result = growth_info if any(v for v in growth_info.values()) else None
        _GROWTH_CACHE[cellosaurus_id] = result
        time.sleep(0.2)
        return result

    except Exception as exc:
        print(f"[growth_properties] Cellosaurus lookup "
              f"failed for {cellosaurus_id}: {exc}")
        _GROWTH_CACHE[cellosaurus_id] = None
        return None


def format_growth_properties(growth_info: dict | None) -> str:
    if not growth_info:
        return "No growth/culture property data available."
    parts = []
    if growth_info.get("category"):
        parts.append(f"Cell type category: {growth_info['category']}")
    if growth_info.get("cell_type"):
        parts.append(f"Cell type: {growth_info['cell_type']}")
    if growth_info.get("doubling_time"):
        dt = growth_info["doubling_time"]
        if dt.get("min") and dt.get("max"):
            parts.append(f"Doubling time: {dt['min']}–{dt['max']} {dt['unit']}s")
        elif dt.get("min"):
            parts.append(f"Doubling time: {dt['min']} {dt['unit']}s")
    if growth_info.get("karyotype"):
        parts.append(f"Karyotype: {growth_info['karyotype'][:120]}")
    return "; ".join(parts) if parts else "No growth/culture property data available."

