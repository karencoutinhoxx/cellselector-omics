import re
import time
import warnings

from Bio import Entrez

Entrez.email = "cellselector@bristol.ac.uk"

# NCBI allows 3 req/s without an API key; sleep between calls to stay within limit
_RATE_SLEEP = 0.34


def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """
    Search PubMed for query and return structured paper metadata.

    Returns a list of dicts, each with:
        pmid, title, authors, journal, year, abstract, url
    Returns [] (with a warning) if PubMed is unreachable or the query fails.
    """
    try:
        handle = Entrez.esearch(db="pubmed", term=query,
                                retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        pmids  = record.get("IdList", [])
    except Exception as exc:
        warnings.warn(f"[pubmed] esearch failed: {exc}")
        return []

    if not pmids:
        return []

    time.sleep(_RATE_SLEEP)

    try:
        handle  = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
    except Exception as exc:
        warnings.warn(f"[pubmed] efetch failed: {exc}")
        return []

    # Normalise: records may be a list or a dict keyed by "PubmedArticle"
    if isinstance(records, list):
        articles = records
    elif isinstance(records, dict) and "PubmedArticle" in records:
        articles = records["PubmedArticle"]
    else:
        articles = []

    results = []
    for art in articles:
        parsed = _parse_article(art)
        if parsed:
            results.append(parsed)
    return results


def _parse_article(article: dict) -> dict | None:
    """Extract structured metadata from a single PubmedArticle record."""
    try:
        mc   = article["MedlineCitation"]
        pmid = str(mc["PMID"])
        art  = mc["Article"]

        title = re.sub(r'<[^>]+>', '', str(art.get("ArticleTitle", "Unknown title"))).strip()

        # Authors: first author Last name + et al.
        authors_raw = art.get("AuthorList", [])
        if authors_raw:
            first = authors_raw[0]
            first_name = first.get("LastName") or first.get("CollectiveName", "Unknown")
            authors = f"{first_name} et al." if len(authors_raw) > 1 else str(first_name)
        else:
            authors = "Unknown"

        journal = str(art["Journal"].get("Title", "Unknown journal"))

        # Year: prefer Year, fall back to MedlineDate first 4 chars
        pub_date   = art["Journal"]["JournalIssue"]["PubDate"]
        year_raw   = pub_date.get("Year") or pub_date.get("MedlineDate", "0")[:4]
        try:
            year = int(str(year_raw)[:4])
        except (ValueError, TypeError):
            year = 0

        # Abstract: can be a plain string or a list of labelled sections
        abstract_raw = art.get("Abstract", {}).get("AbstractText", "")
        if isinstance(abstract_raw, list):
            abstract = " ".join(str(a) for a in abstract_raw)
        else:
            abstract = str(abstract_raw)
        abstract = abstract[:200].strip()

        return {
            "pmid":     pmid,
            "title":    title,
            "authors":  authors,
            "journal":  journal,
            "year":     year,
            "abstract": abstract,
            "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    except Exception:
        return None


def get_cell_line_literature(
    gene: str,
    cell_line_name: str,
    disease: str | None = None,
) -> list[dict]:
    """
    Run three targeted PubMed searches and return up to 5 unique papers.

    Search 1: specific cell line + gene expression
    Search 2: gene + disease + cell line model  (skipped if disease is None)
    Search 3: gene + cancer expression (landmark papers)

    Papers are deduplicated by PMID across all searches.
    """
    queries: list[tuple[str, int]] = [
        (
            f"{cell_line_name}[Title/Abstract] AND "
            f"{gene}[Title/Abstract] AND "
            f"expression[Title/Abstract]",
            3,
        ),
    ]

    if disease:
        queries.append((
            f"{gene}[Title/Abstract] AND "
            f"{disease}[Title/Abstract] AND "
            f"cell line[Title/Abstract]",
            3,
        ))

    queries.append((
        f"{gene}[Title/Abstract] AND "
        f"cancer[Title/Abstract] AND "
        f"expression[Title/Abstract]",
        2,
    ))

    seen_pmids: set[str] = set()
    combined: list[dict] = []

    for query, max_r in queries:
        if len(combined) >= 5:
            break
        time.sleep(_RATE_SLEEP)
        papers = search_pubmed(query, max_results=max_r)
        for p in papers:
            if p["pmid"] not in seen_pmids:
                seen_pmids.add(p["pmid"])
                combined.append(p)
                if len(combined) >= 5:
                    break

    return combined


def format_citations(papers: list[dict]) -> str:
    """
    Format a list of paper dicts into a numbered citation block for the LLM.

    Example output:
        SUPPORTING LITERATURE:
        [1] Smith et al. (2019) - Title here.
            Journal Name. PMID: 12345678
            Abstract: First 150 chars...
            URL: https://pubmed.ncbi.nlm.nih.gov/12345678/
    """
    if not papers:
        return "SUPPORTING LITERATURE:\n  (no relevant papers found)"

    lines = ["SUPPORTING LITERATURE:"]
    for i, p in enumerate(papers, 1):
        abstract_snippet = (p["abstract"][:150] + "...") if len(p["abstract"]) > 150 else p["abstract"]
        lines.append(
            f"[{i}] {p['authors']} ({p['year']}) - {p['title']}\n"
            f"    {p['journal']}. PMID: {p['pmid']}\n"
            f"    Abstract: {abstract_snippet}\n"
            f"    URL: {p['url']}"
        )
    return "\n".join(lines)

